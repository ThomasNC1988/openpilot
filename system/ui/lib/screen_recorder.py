import queue
import threading
import time

import cv2
import numpy as np
from msgq.visionipc import VisionIpcServer, VisionStreamType

from openpilot.common.swaglog import cloudlog
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

# separate VisionIpcServer name (not "camerad") so this doesn't collide with the real
# road camera stream; encoderd is taught to connect to a per-camera vipc_name
SCREEN_VIPC_NAME = "screenrecordd"
SCREEN_STREAM_TYPE = VisionStreamType.VISION_STREAM_ROAD
# kept low and independent of the UI's render fps -- this is a debug/reference recording,
# not a driving-critical stream, and the C3X has little CPU headroom to spare for it
SCREEN_FPS = 10
# recorded at a fraction of native UI resolution to keep the per-frame conversion cost small;
# a prior full-resolution, numpy-only version overloaded the device badly enough to make the
# UI unresponsive
RECORD_WIDTH = 640
MAX_CONSECUTIVE_FAILURES = 5


def _target_size(source_width: int, source_height: int) -> tuple[int, int]:
  width = min(RECORD_WIDTH, source_width)
  height = round(source_height * (width / source_width))
  # NV12 4:2:0 subsampling and VENUS alignment both require even dimensions
  width -= width % 2
  height -= height % 2
  return max(width, 2), max(height, 2)


def _rgba_to_nv12(rgba: np.ndarray, out_width: int, out_height: int, stride: int, y_height: int,
                   uv_height: int, uv_offset: int, buffer_size: int) -> bytes:
  """Downscale + convert an (H, W, 4) RGBA array into a VisionBuf-compatible NV12 buffer.

  Must match the exact VENUS layout system/camerad/cameras/nv12_info.py computes (stride/scanline
  alignment, uv_offset, total buffer_size) -- VisionIpcServer.send() asserts the byte length matches
  exactly what create_buffers_with_sizes() allocated, and the on-device hardware encoder reads the
  buffer directly using this layout. Uses cv2's native color conversion (vs. hand-rolled numpy math)
  since it's dramatically cheaper on-device.
  """
  if rgba.shape[1] != out_width or rgba.shape[0] != out_height:
    rgba = cv2.resize(rgba, (out_width, out_height), interpolation=cv2.INTER_AREA)

  yuv_i420 = cv2.cvtColor(rgba, cv2.COLOR_RGBA2YUV_I420)  # shape (out_height * 3 // 2, out_width)
  y = yuv_i420[:out_height, :]
  u = yuv_i420[out_height: out_height + out_height // 4].reshape(out_height // 2, out_width // 2)
  v = yuv_i420[out_height + out_height // 4: out_height + out_height // 2].reshape(out_height // 2, out_width // 2)

  uv = np.empty((out_height // 2, out_width), dtype=np.uint8)
  uv[:, 0::2] = u
  uv[:, 1::2] = v

  buf = np.zeros(buffer_size, dtype=np.uint8)
  y_plane = buf[: stride * y_height].reshape(y_height, stride)
  y_plane[:out_height, :out_width] = y
  uv_plane = buf[uv_offset: uv_offset + stride * uv_height].reshape(uv_height, stride)
  uv_plane[: uv.shape[0], : uv.shape[1]] = uv

  return buf.tobytes()


class ScreenRecorder:
  """Publishes downscaled UI frames as an NV12 VisionIPC stream so encoderd can hardware-encode them."""

  def __init__(self, source_width: int, source_height: int):
    self._source_width = source_width
    self._source_height = source_height
    self._width, self._height = _target_size(source_width, source_height)
    self._stride, self._y_height, self._uv_height, self._buffer_size = get_nv12_info(self._width, self._height)
    self._uv_offset = self._stride * self._y_height

    self._server = VisionIpcServer(SCREEN_VIPC_NAME)
    self._server.create_buffers_with_sizes(
      SCREEN_STREAM_TYPE, 4, self._width, self._height, self._buffer_size, self._stride, self._uv_offset
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
    if self._stop_event.is_set():
      return False
    now = time.monotonic()
    if now - self._last_capture_time < 1.0 / SCREEN_FPS:
      return False
    self._last_capture_time = now
    return True

  def push_frame(self, rgba_bytes: bytes):
    """Hand off raw RGBA bytes (at source_width x source_height) for background conversion/publish.
    Never blocks the caller."""
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
    consecutive_failures = 0
    while not self._stop_event.is_set():
      try:
        data = self._queue.get(timeout=1.0)
      except queue.Empty:
        continue

      try:
        rgba = np.frombuffer(data, dtype=np.uint8).reshape(self._source_height, self._source_width, 4)
        rgba = np.flipud(rgba)  # raylib's screen/texture readback is bottom-up
        nv12 = _rgba_to_nv12(rgba, self._width, self._height, self._stride, self._y_height,
                              self._uv_height, self._uv_offset, self._buffer_size)
        eof = int(self._frame_id * (1e9 / SCREEN_FPS))
        self._server.send(SCREEN_STREAM_TYPE, nv12, self._frame_id, eof, eof)
        self._frame_id += 1
        consecutive_failures = 0
      except Exception as e:
        consecutive_failures += 1
        cloudlog.error(f"screen recorder failed to encode/publish frame ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}")
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
          cloudlog.error("screen recorder: too many consecutive failures, disabling")
          self._stop_event.set()
          break

  def close(self):
    self._stop_event.set()
    if self._thread.is_alive():
      self._thread.join(timeout=5)
