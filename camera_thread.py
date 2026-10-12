"""
camera_thread.py — feeds frames into state.frame from one of three sources,
selected by CAMERA_SOURCE in config:

  "picam" → Raspberry Pi CSI camera via libcamera (rpicam-vid subprocess)
  "usb"   → USB / V4L2 webcam via OpenCV
  "esp32" → ESP32-CAM MJPEG stream over WiFi
"""

import cv2
import threading
import time
import subprocess
import numpy as np
from config import (CAMERA_SOURCE, CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT,
                    FPS, ESP32_STREAM,
                    PICAM_WIDTH, PICAM_HEIGHT, PICAM_FPS, PICAM_HFLIP, PICAM_VFLIP)
from shared_state import state
import console


def _read_exact(stream, n):
    """Read exactly n bytes from a pipe, or None if the stream ended."""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _picam_loop():
    """Raspberry Pi CSI camera via rpicam-vid. Reads raw YUV420 frames from the
    subprocess's stdout and converts them to BGR for the vision pipeline.

    A subprocess is used on purpose: picamera2/libcamera's Python bindings are
    built for the system Python, but Luna's venv is a different version, so they
    can't be imported here — rpicam-vid sidesteps that entirely."""
    w, h, fps = PICAM_WIDTH, PICAM_HEIGHT, PICAM_FPS
    frame_bytes = w * h * 3 // 2   # YUV420 = 1.5 bytes/pixel
    cmd = ["rpicam-vid", "-t", "0", "--nopreview",
           "--width", str(w), "--height", str(h),
           "--framerate", str(fps), "--codec", "yuv420", "--flush", "-o", "-"]
    if PICAM_HFLIP:
        cmd.append("--hflip")
    if PICAM_VFLIP:
        cmd.append("--vflip")

    console.detail(f"[camera] Starting Pi CSI camera (rpicam-vid) {w}x{h} @ {fps}fps")
    help_printed = False

    while True:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, bufsize=0)
        except FileNotFoundError:
            if not help_printed:
                help_printed = True
                print("[camera] rpicam-vid not found — no libcamera? vision disabled, "
                      "retrying every 5s")
            with state.lock:
                state.camera_ok = False
            time.sleep(5.0)
            continue

        with state.lock:
            state.camera_ok = True

        try:
            while True:
                buf = _read_exact(proc.stdout, frame_bytes)
                if buf is None:
                    break   # stream ended — reopen below
                yuv   = np.frombuffer(buf, dtype=np.uint8).reshape((h * 3 // 2, w))
                frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                with state.lock:
                    state.frame = frame
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                proc.kill()

        with state.lock:
            state.camera_ok = False
        print("[camera] Pi camera stream ended — restarting in 2s")
        time.sleep(2.0)


def _esp32_loop():
    """Read MJPEG stream from ESP32."""
    console.detail(f"[camera] Connecting to ESP32 stream: {ESP32_STREAM}")

    while True:
        cap = cv2.VideoCapture(ESP32_STREAM)

        if not cap.isOpened():
            print("[camera] ESP32 stream not available — retrying in 3s")
            with state.lock:
                state.camera_ok = False
            time.sleep(3)
            continue

        console.detail("[camera] ESP32 stream connected")
        with state.lock:
            state.camera_ok = True

        while True:
            ret, frame = cap.read()

            if not ret:
                print("[camera] ESP32 stream lost — reconnecting")
                break

            # resize to match expected frame size
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            with state.lock:
                state.frame = frame

        cap.release()
        time.sleep(2)   # wait before reconnecting


def _open_local_camera():
    """Try the configured camera, then fall back to indices 0-2."""
    tried = []
    for cam_id in [CAMERA_ID, 0, 1, 2]:
        if cam_id in tried:
            continue
        tried.append(cam_id)
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS,          FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
            if cam_id != CAMERA_ID:
                print(f"[camera] Camera {CAMERA_ID} unavailable — using {cam_id}")
            return cap
        cap.release()
    return None


def _local_loop():
    """Read from local USB or Pi camera. Retries forever if no camera —
    Luna keeps running (voice + face) without vision."""
    console.detail(f"[camera] Starting local camera {CAMERA_ID}")
    help_printed = False

    while True:
        cap = _open_local_camera()

        if cap is None:
            if not help_printed:
                help_printed = True
                print("[camera] NO CAMERA FOUND — vision disabled, retrying every 5s")
                print("[camera] See README Troubleshooting: 'Camera not found'")
            with state.lock:
                state.camera_ok = False
            time.sleep(5.0)
            continue

        help_printed = False
        with state.lock:
            state.camera_ok = True
        console.detail(f"[camera] Local camera started {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FPS}fps")

        fail_count = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                fail_count += 1
                if fail_count > 60:   # ~3s of failures → reopen device
                    print("[camera] Camera stopped responding — reopening")
                    break
                time.sleep(0.05)
                continue

            fail_count = 0
            with state.lock:
                state.frame = frame

        cap.release()
        time.sleep(2.0)


def start_camera():
    src = (CAMERA_SOURCE or "picam").lower()
    if src == "esp32":
        target = _esp32_loop
    elif src == "usb":
        target = _local_loop
    else:                       # default: Pi CSI camera
        target = _picam_loop
    t = threading.Thread(target=target, daemon=True, name="camera")
    t.start()