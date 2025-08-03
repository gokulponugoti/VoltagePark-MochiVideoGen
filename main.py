import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router  # 👈 assuming routes.py exports APIRouter

# 🪵 Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Starting FastAPI application...")

app = FastAPI(
    title="AI Inference API",
    description="Modular GPU-optimized endpoint for text-to-video and model orchestration",
    version="1.0.0",
)

# 🌐 CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔐 replace with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware configured.")

# 🧩 Include app routes
app.include_router(router, prefix="/api")
logger.info("API routes included.")

# ✅ Health check
@app.get("/health")
def health():
    logger.info("Health check endpoint called.")
    return {"status": "healthy"}

# ✅ GPU status endpoint
@app.get("/gpu-status")
def gpu_status():
    logger.info("GPU status endpoint called.")
    try:
        import torch
        available = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if available else "None"
        logger.info(f"GPU available: {available}, device: {device}")
        return {
            "available": available,
            "device": device
        }
    except Exception as e:
        logger.error(f"Error checking GPU status: {e}")
        return {"error": str(e)}
