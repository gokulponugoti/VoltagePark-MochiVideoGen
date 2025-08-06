import os
import uuid
import time
import redis
import logging
from datetime import datetime
from fastapi import FastAPI, Query
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = 6379
REDIS_DB = 0
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

app = FastAPI(
    title="Mochi Video API",
    version="1.0.0",
    description="Text prompts to video generation"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

QUEUE_ZSET_KEY = "jobs_queue"
ALL_JOBS_ZSET_KEY = "all_jobs"

@app.post("/api/generate-video")
def generate_video(request: GenerateVideoRequest):
    prompt = request.prompt
    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    timestamp = time.time()

    job_data = {
        "id": job_id,
        "prompt": prompt,
        "status": "queued",
        "output_path": "",
        "output_url": "",
        "created_at": now,
        "updated_at": now
    }

    r.hset(f"job:{job_id}", mapping=job_data)

    # Add to queue and all_jobs sorted sets
    r.zadd(QUEUE_ZSET_KEY, {job_id: timestamp})
    r.zadd(ALL_JOBS_ZSET_KEY, {job_id: timestamp})

    logger.info(f"Job {job_id} queued: {prompt}")
    return {"job_id": job_id}

@app.get("/api/job-status/{job_id}")
def check_job(job_id: str):
    job_data = r.hgetall(f"job:{job_id}")
    if not job_data:
        return {"error": "Job not found"}
    return job_data

@app.get("/api/jobs")
def list_jobs(
    limit: int = Query(10, ge=1, le=100, description="Max jobs to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    status: Optional[str] = Query(None, description="Comma-separated list of statuses to filter by")
):
    allowed_statuses = {"queued", "processing", "completed", "error"}
    status_filter = None

    if status:
        status_filter = set(s.strip().lower() for s in status.split(","))
        invalid_statuses = status_filter - allowed_statuses
        if invalid_statuses:
            return {"error": f"Invalid status value(s): {', '.join(invalid_statuses)}. Allowed: {', '.join(allowed_statuses)}"}

    total = r.zcard(ALL_JOBS_ZSET_KEY)
    job_ids = r.zrevrange(ALL_JOBS_ZSET_KEY, offset, offset + limit - 1)

    jobs = []
    for job_id in job_ids:
        data = r.hgetall(f"job:{job_id}")
        if data:
            if not status_filter or data.get("status") in status_filter:
                jobs.append(data)

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "jobs": jobs
    }
