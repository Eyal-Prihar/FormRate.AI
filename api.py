"""
FormRate.ai — FastAPI Server
Wraps the FormRateAnalyzer so the React frontend can POST a video and get a JSON report.

Usage:
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --reload --port 8000

Then open http://localhost:8000 in your browser.
"""

import os
import tempfile
from typing import Literal

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from formrate import FormRateAnalyzer, FormReport

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(title="FormRate.ai API", version="0.1.0")

# ── CORS ──────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://[::]:5500",
        "http://[::1]:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One shared analyzer instance (loads the MediaPipe model once at startup)
analyzer = FormRateAnalyzer(model_complexity=1)


# ── Serve frontend ────────────────────────────────────────────────────────
@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


# ── Health check ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ── Main analysis endpoint ────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(
    video: UploadFile = File(..., description="Video file to analyze"),
    exercise: Literal["squat", "deadlift", "bench"] = Form(..., description="Exercise type"),
):
    if video.content_type and not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file does not appear to be a video.")

    suffix = os.path.splitext(video.filename or "upload.mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        contents = await video.read()
        tmp.write(contents)
        tmp.flush()
        tmp.close()

        report: FormReport = analyzer.analyze(tmp.name, exercise=exercise)
        return JSONResponse(content=_report_to_dict(report))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


# ── Serializer ────────────────────────────────────────────────────────────
def _report_to_dict(report: FormReport) -> dict:
    return {
        "exercise": report.exercise,
        "video_path": os.path.basename(report.video_path),
        "total_reps": report.total_reps,
        "overall_score": report.overall_score,
        "overall_feedback": report.overall_feedback,
        "rep_scores": [
            {
                "exercise": rs.exercise,
                "rep_number": rs.rep_number,
                "score": rs.score,
                "feedback": rs.feedback,
                "raw_angles": {k: round(v, 2) if v is not None else None for k, v in rs.raw_angles.items()},
                "categories": [
                    {
                        "name": cat.name,
                        "score": round(cat.score, 2),
                        "max_score": cat.max_score,
                        "feedback": cat.feedback,
                        "checks": [
                            {
                                "name": c.name,
                                "passed": c.passed,
                                "deduction": c.deduction,
                                "feedback": c.feedback,
                                "value": round(c.value, 2) if c.value is not None else None,
                            }
                            for c in cat.checks
                        ],
                    }
                    for cat in (rs.categories if hasattr(rs, "categories") else [])
                ],
                "checks": [
                    {
                        "name": c.name,
                        "passed": c.passed,
                        "deduction": c.deduction,
                        "feedback": c.feedback,
                        "value": round(c.value, 2) if c.value is not None else None,
                    }
                    for c in rs.checks
                ],
            }
            for rs in report.rep_scores
        ],
    }
