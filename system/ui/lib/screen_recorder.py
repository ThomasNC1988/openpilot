import queue
import threading
import time

import numpy as np
from msgq.visionipc import VisionIpcServer, VisionStreamType

from openpilot.common.swaglog import cloudlog
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

# separate VisionIpcServer name (not "camerad") so this doesn't collide with the real
# road camera stream; encoderd is taught to connect to a per-camera vipc_name
SCREEN_VIPC_NAME = "screenrecordd"
SCREEN_STREAM_TYPE = VisionStreamType.VISION_STREAM_ROAD
# matches MAIN_FPS in system/loggerd/loggerd.h -- encoderd's segment rotation length
# is computed in units of MAIN_FPS regardless of the source's actual frame rate
SCREEN_FPS = 20


def _rgba_to_nv12(rgba: np.ndarray, stride: int, y_height: int, uv_height: int, uv_offset: int, buffer_size: int) -> bytes:
  """Convert an (H, W, 4) RGBA array into a VisionBuf-compatible NV12 buffer.

  Must match the exact VENUS layout system/camerad/cameras/nv12_info.py computes (stride/scanline
  alignment, uv_offset, total buffer_size) -- VisionIpcServer.send() asserts the byte length matches
  exactly what create_buffers_with_sizes() allocated, and the on-device hardware encoder reads the
  buffer directly using this layout.
  """
  height, width = rgba.shape[:2]
  r = rgba[:, :, 0].astype(np.int32)
  g = rgba[:, :, 1].astype(np.int32)
  b = rgba[:, :, 2].astype(np.int32)

  y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16
  u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128
  v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128

  y = np.clip(y, 0, 255).astype(np.uint8)
  u = np.clip(u[0::2, 0::2], 0, 255).astype(np.uint8)
  v = np.clip(v[0::2, 0::2], 0, 255).astype(np.uint8)

  uv = np.empty((u.shape[0], u.shape[1] * 2), dtype=np.uint8)
  uv[:, 0::2] = u
  uv[:, 1::2] = v

  buf = np.zeros(buffer_size, dtype=np.uint8)
  y_plane = buf[: stride * y_height].reshape(y_height, stride)
  y_plane[:height, :width] = y
  uv_plane = buf[uv_offset : uv_offset + stride * uv_height].reshape(uv_height, stride)
  uv_plane[: uv.shape[0], : uv.shape[1]] = uv

  return buf.tobytes()


class ScreenRecorder:
  """Publishes UI frames as an NV12 VisionIPC stream so encoderd can hardware-encode them."""

  def __init__(self, width: int, height: int):
    self._width = width
    self._height = height
    self._stride, self._y_height, self._uv_height, self._buffer_size = get_nv12_info(width, height)
    self._uv_offset = self._stride * self._y_height

    self._server = VisionIpcServer(SCREEN_VIPC_NAME)
    self._server.create_buffers_with_sizes(
      SCREEN_STREAM_TYPE, 4, width, height, self._buffer_size, self._stride, self._uv_offset
    )
    self._server.start_listener()

    self._frame_id = 0
    self._last_capture_time = 0.0
    self._queue: queue.Queue = queue.Queue(maxsize=2)
    self._stop_event = threading.Event()
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()

  def should_capture(self) -> bool:
    """Call before doing an (expensive) GPU readback -- throttles capture to SCREEN_FPS."""
    now = time.monotonic()
    if now - self._last_capture_time < 1.0 / SCREEN_FPS:
      return False
    self._last_capture_time = now
    return True

  def push_frame(self, rgba_bytes: bytes):
    """Hand off raw RGBA bytes for background conversion/publish. Never blocks the caller."""
    try:
      self._queue.put_nowait(rgba_bytes)
    except queue.Full:
      # drop the stale frame in favor of the fresher one; never block the render loop
      try:
        self._queue.get_nowait()
      except queue.Empty:
        pass
      try:
        self._queue.put_nowait(rgba_bytes)
      except queue.Full:
        pass

  def _run(self):
    while not self._stop_event.is_set():
      try:
        data = self._queue.get(timeout=1.0)
      except queue.Empty:
        continue

      try:
        rgba = np.frombuffer(data, dtype=np.uint8).reshape(self._height, self._width, 4)
        rgba = np.flipud(rgba)  # raylib's texture readback is bottom-up
        nv12 = _rgba_to_nv12(rgba, self._stride, self._y_height, self._uv_height, self._uv_offset, self._buffer_size)
        eof = int(self._frame_id * (1e9 / SCREEN_FPS))
        self._server.send(SCREEN_STREAM_TYPE, nv12, self._frame_id, eof, eof)
        self._frame_id += 1
      except Exception as e:
        cloudlog.error(f"screen recorder failed to encode/publish frame: {e}")

  def close(self):
    self._stop_event.set()
    if self._thread.is_alive():
      self._thread.join(timeout=5)
