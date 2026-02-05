"""
FastAPI service for tax document classification.
Upload a PDF and get back classified form types per page range.

Weights: load from MODEL_PATH env (or .env), default "models/semantic_classifier.pkl".
Train and save first: python main.py --mode train
"""
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.models import HybridClassifier, SemanticClassifier
from src.pipeline import DocumentClassifierPipeline
from src.schemas import ClassificationResult, HealthResponse
from src.utils import setup_logger

logger = setup_logger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/semantic_classifier.pkl")


def get_pipeline() -> DocumentClassifierPipeline:
    path = Path(MODEL_PATH)
    if path.exists():
        semantic = SemanticClassifier.load(str(path))
        logger.info("Loaded semantic classifier from %s", MODEL_PATH)
    else:
        logger.warning("No weights at %s; using heuristics only.", MODEL_PATH)
        semantic = SemanticClassifier()
    hybrid = HybridClassifier(semantic_classifier=semantic)
    return DocumentClassifierPipeline(hybrid)


_pipeline: Optional[DocumentClassifierPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = get_pipeline()
    yield
    _pipeline = None


STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Tax Document Classifier API",
    description="Upload a PDF to get classified form types (e.g. Form 1040, Schedule A) per page range.",
    lifespan=lifespan,
)


@app.get("/")
def index():
    """Serve the minimal frontend (PDF upload, form list, JSON copy)."""
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_path=MODEL_PATH)


@app.post("/classify", response_model=ClassificationResult)
async def classify(file: UploadFile = File(..., description="PDF file to classify")):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload must be a PDF file.")
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Classifier not loaded.")
    try:
        contents = await file.read()
    except Exception as e:
        logger.exception("Failed to read upload")
        raise HTTPException(status_code=400, detail="Failed to read file.") from e
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        try:
            tmp.write(contents)
            tmp.flush()
            result = _pipeline.process_file(tmp.name)
            return ClassificationResult.model_validate(result)
        except Exception as e:
            logger.exception("Classification failed")
            raise HTTPException(status_code=500, detail="Classification failed.") from e
        finally:
            Path(tmp.name).unlink(missing_ok=True)
