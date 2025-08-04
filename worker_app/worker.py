import os
import redis
import time
import logging
from model_runner import run_inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = 6379
REDIS_DB = 0

MOCHI_WEIGHTS_DIR = os.getenv("MOCHI_WEIGHTS_DIR", "/weights")
OUTPUT_DIR = "/worker/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

while True:
    job_id = r.lpop("video_jobs")
    if not job_id:
        time.sleep(2)
        continue

    job_key = f"job:{job_id}"
    job = r.hgetall(job_key)

    if not job:
        logger.warning(f"Job ID {job_id} not found.")
        continue

    prompt = job["prompt"]
    output_path = f"{OUTPUT_DIR}/{prompt.replace(' ', '_')}.mp4"

    try:
        r.hset(job_key, mapping={"status": "processing"})
        logger.info(f"Processing job {job_id}: {prompt}")
        run_inference(prompt, output_path)
        r.hset(job_key, mapping={"status": "completed", "output_path": output_path})
        logger.info(f"Job {job_id} completed and saved to {output_path}")
    except Exception as e:
        logger.exception(f"Job {job_id} failed.")
        r.hset(job_key, mapping={"status": "error", "output_path": ""})
