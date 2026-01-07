import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import db
from app.pipeline import InferencePipeline
from app.routes import devices, feedback, infer, zones

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="ValveLens", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    app.state.pipeline = InferencePipeline()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(infer.router)
app.include_router(zones.router)
app.include_router(devices.router)
app.include_router(feedback.router)
