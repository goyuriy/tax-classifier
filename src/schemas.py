"""Pydantic models for API and pipeline data contracts."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    model_path: str


class FormRange(BaseModel):
    """A contiguous range of pages classified as one form type."""

    document_type: str = Field(..., description="Form identifier (e.g. 1040f, f1040sa)")
    start_page: int = Field(..., ge=1, description="First page of this form")
    end_page: int = Field(..., ge=1, description="Last page of this form")


class ClassificationResult(BaseModel):
    """Response shape for document classification."""

    forms: list[FormRange] = Field(..., description="Detected form ranges in page order")
