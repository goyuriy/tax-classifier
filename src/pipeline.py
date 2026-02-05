import json
import time
from pathlib import Path
from typing import Dict, Any
from src.loader import PDFLoader
from src.models import HybridClassifier
from src.utils import setup_logger

logger = setup_logger(__name__)


def _ms(seconds: float) -> float:
    return seconds * 1000.0


class DocumentClassifierPipeline:
    def __init__(self, classifier: HybridClassifier):
        self.loader = PDFLoader()
        self.classifier = classifier

    def process_file(self, file_path: str) -> Dict[str, Any]:
        logger.info("Processing file: %s", file_path)
        pages = self.loader.load(file_path)
        num_pages = len(pages)

        forms = []
        current_form = None

        t0 = time.perf_counter()
        for page in pages:
            # Predict
            if page.is_scanned:
                # In a real system, we'd use OCR here.
                # For now, we might skip or mark as unknown/scanned
                label = "scanned_document" 
            else:
                prediction = self.classifier.predict(page.text)
                label = prediction["label"]
            
            # Grouping logic
            if current_form is None:
                current_form = {
                    "document_type": label,
                    "start_page": page.page_number,
                    "end_page": page.page_number
                }
            else:
                # If same label, extend. 
                # Note: This merges two distinct 1040 forms into one if they are adjacent.
                # Real logic might need to check for "start of form" signals to split same-type forms.
                if label == current_form["document_type"]:
                    current_form["end_page"] = page.page_number
                else:
                    forms.append(current_form)
                    current_form = {
                        "document_type": label,
                        "start_page": page.page_number,
                        "end_page": page.page_number
                    }
        
        if current_form:
            forms.append(current_form)

        elapsed = time.perf_counter() - t0
        elapsed_ms = _ms(elapsed)
        per_page_ms = elapsed_ms / num_pages if num_pages else 0
        logger.info(
            "Inference: %.2f ms total (%d pages), %.2f ms/page",
            elapsed_ms, num_pages, per_page_ms,
        )
        return {"forms": forms}

    def process_directory(self, input_dir: str, output_dir: str):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for pdf_file in input_path.glob("*.pdf"):
            result = self.process_file(str(pdf_file))
            
            output_file = output_path / f"{pdf_file.stem}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved result to {output_file}")
