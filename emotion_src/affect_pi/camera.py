from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


class Camera(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def release(self) -> None: ...


@dataclass
class WebcamCamera:
    camera_index: int = 0
    width: int = 640
    height: int = 480

    def __post_init__(self) -> None:
        self._cap = cv2.VideoCapture(self.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {self.camera_index}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._cap.read()
        return bool(ok), frame if ok else None

    def release(self) -> None:
        self._cap.release()


@dataclass
class PiCamera2Camera:
    width: int = 640
    height: int = 480

    def __post_init__(self) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "picamera2 is not installed. On Raspberry Pi, install camera support "
                "and run: pip install -e .[pi]"
            ) from exc

        self._picam2 = Picamera2()
        config = self._picam2.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        )
        self._picam2.configure(config)
        self._picam2.start()

    def read(self) -> tuple[bool, np.ndarray | None]:
        rgb = self._picam2.capture_array()
        if rgb is None:
            return False, None
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return True, bgr

    def release(self) -> None:
        self._picam2.stop()


def make_camera(kind: str, camera_index: int, width: int, height: int) -> Camera:
    kind = kind.lower().strip()
    if kind == "webcam":
        return WebcamCamera(camera_index=camera_index, width=width, height=height)
    if kind == "picamera2":
        return PiCamera2Camera(width=width, height=height)
    raise ValueError(f"Unknown camera kind: {kind}")
