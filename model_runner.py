import os
import torch
import logging
from genmo.mochi_preview.pipelines import (
    DecoderModelFactory, DitModelFactory, MochiSingleGPUPipeline,
    T5ModelFactory, linear_quadratic_schedule
)
from genmo.lib.utils import save_video

# 🪵 Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MOCHI_WEIGHTS_DIR", "/home/ubuntu/mochi_weights")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Using device: {DEVICE}")
logger.info(f"Loading model weights from: {MODEL_PATH}")

try:
    pipeline = MochiSingleGPUPipeline(
        text_encoder_factory=T5ModelFactory(),
        dit_factory=DitModelFactory(
            model_path=f"{MODEL_PATH}/dit.safetensors", model_dtype="bf16"
        ),
        decoder_factory=DecoderModelFactory(
            model_path=f"{MODEL_PATH}/vae.safetensors",
        ),
        cpu_offload=True,
        decode_type="tiled_full"
    )
    logger.info("Pipeline initialized successfully.")
except Exception as e:
    logger.exception("Failed to initialize pipeline.")
    raise e

def run_inference(prompt: str, output_path: str):
    logger.info(f"Running inference for prompt: {prompt}")
    try:
        video = pipeline(
            height=480, width=848, num_frames=31,
            num_inference_steps=64,
            sigma_schedule=linear_quadratic_schedule(64, 0.025),
            cfg_schedule=[4.5]*64,
            batch_cfg=False,
            prompt=prompt, negative_prompt="", seed=42
        )
        save_video(video[0], output_path)
        logger.info(f"Video generated and saved to: {output_path}")
    except Exception as e:
        logger.exception("Error during inference.")
        raise e
