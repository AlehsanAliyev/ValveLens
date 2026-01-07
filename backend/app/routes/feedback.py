from typing import Dict

from fastapi import APIRouter
from pydantic import BaseModel

from app import db

router = APIRouter()


class FeedbackRequest(BaseModel):
    obs_id: str
    feedback_type: str
    data_json: Dict


@router.post("/feedback")
def feedback(payload: FeedbackRequest) -> dict:
    fb_id = db.insert_feedback(payload.obs_id, payload.feedback_type, payload.data_json)
    return {"feedback_id": fb_id}
