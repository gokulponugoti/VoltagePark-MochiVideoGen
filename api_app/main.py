import os
import uuid
import redis
import logging
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = 6379
REDIS_DB = 0

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

app = FastAPI(
    title="Mochi Video API",
    version="1.0.0",
    description="Queue text prompts for video generation"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    prompt: str

@app.post("/api/generate-video")
def generate_video(prompt: str = Query(..., description="Text prompt for video")):
    job_id = str(uuid.uuid4())
    job_data = {
        "id": job_id,
        "prompt": prompt,
        "status": "queued",
        "output_path": ""
    }
    r.hset(f"job:{job_id}", mapping=job_data)
    r.rpush("video_jobs", job_id)
    logger.info(f"Job {job_id} queued for prompt: {prompt}")
    return {"job_id": job_id}

@app.get("/api/job-status/{job_id}")
def check_job(job_id: str):
    job_data = r.hgetall(f"job:{job_id}")
    if not job_data:
        return {"error": "Job not found"}
    return job_data
