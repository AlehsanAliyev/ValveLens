from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class StrictBase(BaseModel):
    class Config:
        extra = "forbid"


class InputInfo(StrictBase):
    type: Literal["webcam", "video", "image"]
    source: str
    frame_index: Optional[int] = None


class QualityInfo(StrictBase):
    blur_score: float
    brightness: float
    glare_score: float
    is_low_light: bool
    is_blurry: bool


class ZoneCandidate(StrictBase):
    zone_id: str
    zone_name: Optional[str] = None
    score: float


class ZoneTop1(StrictBase):
    zone_id: str
    zone_name: Optional[str] = None
    score: float


class ZoneInfo(StrictBase):
    candidates: List[ZoneCandidate]
    top1: Optional[ZoneTop1] = None


class BBox(StrictBase):
    x1: int
    y1: int
    x2: int
    y2: int


class RoiInfo(StrictBase):
    crop_path: Optional[str] = None
    mask_path: Optional[str] = None


class OcrInfo(StrictBase):
    text: Optional[str] = None
    conf: Optional[float] = None
    boxes: Optional[List[Any]] = None


class ReIdMatch(StrictBase):
    device_id: str
    score: float


class ReIdInfo(StrictBase):
    embedding_type: str
    top_matches: List[ReIdMatch]


class FusedInfo(StrictBase):
    device_id: Optional[str] = None
    final_score: float
    score_breakdown: Dict[str, Any]


class DetectionInfo(StrictBase):
    det_id: str
    cls: str
    conf: float
    bbox: BBox
    track_id: Optional[str] = None
    roi: RoiInfo
    ocr: Optional[OcrInfo] = None
    reid: ReIdInfo
    fused: FusedInfo


class SelectedDevice(StrictBase):
    device_id: str
    score: float


class BinaryQuestion(StrictBase):
    q: str
    options: List[str]


class UiHints(StrictBase):
    highlight_det_id: Optional[str] = None
    suggested_moves: Optional[List[str]] = None
    binary_question: Optional[BinaryQuestion] = None


class DecisionInfo(StrictBase):
    status: Literal["ACCEPTED", "UNCERTAIN"]
    selected_device: Optional[SelectedDevice] = None
    action: Literal[
        "NONE",
        "ASK_VIEWPOINT",
        "ASK_BINARY",
        "ASK_TAP",
        "ASK_WIDER_VIEW",
    ]
    message: str
    ui_hints: UiHints


class InferenceResponse(StrictBase):
    request_id: str
    timestamp: str
    input: InputInfo
    quality: QualityInfo
    zone: ZoneInfo
    detections: List[DetectionInfo]
    decision: DecisionInfo
