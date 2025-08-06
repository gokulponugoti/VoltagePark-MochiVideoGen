import os
import redis
import time
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import boto3
from model_runner import run_inference

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis config
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = 6379
REDIS_DB = 0
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

# Check Redis connection at startup
try:
    r.ping()
    logger.info("Connected to Redis successfully.")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    exit(1)

# AWS S3 config
S3_BUCKET = os.getenv("S3_BUCKET", "my-mochi-results")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
s3 = boto3.client("s3", region_name=AWS_REGION)

# Output dir on mounted NVMe
OUTPUT_DIR = "/mnt/nvme/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Redis key names
QUEUE_ZSET_KEY = "jobs_queue"
ALL_JOBS_ZSET_KEY = "all_jobs"


def upload_to_s3(file_path, job_id):
    s3_key = f"mochi-results/{job_id}.mp4"
    s3.upload_file(file_path, S3_BUCKET, s3_key)
    url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': S3_BUCKET, 'Key': s3_key},
        ExpiresIn=1200  # 20 mins
    )
    return url


def process_job(job_id):
    job_key = f"job:{job_id}"
    job = r.hgetall(job_key)
    if not job:
        logger.warning(f"Job {job_id} not found.")
        return

    if job.get("status") != "queued":
        logger.warning(f"Job {job_id} skipped (status: {job.get('status')})")
        r.zrem(QUEUE_ZSET_KEY, job_id)
        return

    prompt = job["prompt"]
    output_filename = f"{prompt.replace(' ', '_')}_{job_id}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    now = datetime.utcnow().isoformat()

    try:
        r.hset(job_key, mapping={"status": "processing", "updated_at": now})
        r.zrem(QUEUE_ZSET_KEY, job_id)
        logger.info(f"Processing job {job_id}: {prompt}")
        run_inference(prompt, output_path)

        s3_url = upload_to_s3(output_path, job_id)
        os.remove(output_path)  # Cleanup local file after upload

        now = datetime.utcnow().isoformat()
        r.hset(job_key, mapping={
            "status": "completed",
            "output_url": s3_url,
            "updated_at": now
        })
        logger.info(f"Job {job_id} completed: {s3_url}")

    except Exception:
        logger.exception(f"Job {job_id} failed.")
        now = datetime.utcnow().isoformat()
        r.hset(job_key, mapping={
            "status": "error",
            "updated_at": now,
            "output_url": ""
        })


def worker_loop(max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()
        while True:
            done = {fut for fut in futures if fut.done()}
            futures.difference_update(done)
            for fut in done:
                try:
                    fut.result()
                except Exception:
                    logger.exception("Exception in worker thread")

            if len(futures) < max_workers:
                try:
                    logger.debug(f"Polling queue '{QUEUE_ZSET_KEY}' for jobs...")
                    items = r.zrange(QUEUE_ZSET_KEY, 0, 0)
                    logger.debug(f"Queue returned items: {items}")
                except Exception as e:
                    logger.error(f"Error fetching from Redis queue: {e}")
                    time.sleep(5)
                    continue

                if items:
                    job_id = items[0]
                    logger.info(f"Dequeued job {job_id}")
                    futures.add(executor.submit(process_job, job_id))
                else:
                    time.sleep(2)
            else:
                time.sleep(1)


if __name__ == "__main__":
    worker_loop(max_workers=8)
