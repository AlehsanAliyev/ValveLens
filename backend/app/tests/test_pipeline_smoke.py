import numpy as np

from app import db
from app.pipeline import InferencePipeline


def test_pipeline_smoke() -> None:
    db.init_db()
    pipeline = InferencePipeline()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    response = pipeline.process_frame(
        frame, input_type="image", source="test.jpg", frame_index=0
    )
    assert response.request_id
    assert response.input.type == "image"
