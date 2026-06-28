from enum import Enum


class VisionStatus(str, Enum):
    FACE_VISIBLE = "FACE_VISIBLE"
    BODY_VISIBLE = "BODY_VISIBLE"
    CANT_SEE = "CANT_SEE"
