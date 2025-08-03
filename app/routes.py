import logging
from fastapi import APIRouter, HTTPException
from model_runner import run_inference
import os

# 🪵 Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/generate-video")
def generate_video(prompt: str):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{prompt.replace(' ', '_')}.mp4")
    
    logger.info(f"Received video generation request for prompt: '{prompt}'")
    
    try:
        run_inference(prompt, output_path)
        logger.info(f"Video generation completed: {output_path}")
        return {"video_path": output_path}
    except Exception as e:
        logger.exception("Video generation failed.")
        raise HTTPException(status_code=500, detail="Video generation failed. Check logs for details.")
