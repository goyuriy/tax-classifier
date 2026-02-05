import json
from pathlib import Path
from typing import List, Tuple
from src.loader import PDFLoader
from src.utils import setup_logger

logger = setup_logger(__name__)


def load_training_data(data_dir: str, max_chars: int = 50, include_background: bool = False) -> Tuple[List[str], List[str]]:
    """
    Load (text, label) for every page.
    If include_background is True, pages without a label in target JSONs are labeled as "other".
    Text is truncated to first max_chars to match inference behavior (header zone only).
    """
    input_dir = Path(data_dir) / "input"
    target_dir = Path(data_dir) / "target"
    loader = PDFLoader()

    X_text = []
    y_labels = []
    total_pages = 0

    for pdf_file in sorted(input_dir.glob("*.pdf")):
        json_file = target_dir / f"{pdf_file.stem}.json"
        
        # If target doesn't exist, all pages are background if include_background=True
        if not json_file.exists():
            if include_background:
                try:
                    pages = loader.load(str(pdf_file))
                    total_pages += len(pages)
                    for page in pages:
                        X_text.append(page.text[:max_chars])
                        y_labels.append("other")
                except Exception as e:
                    logger.error(f"Skipping {pdf_file} due to error: {e}")
            continue

        with open(json_file, "r") as f:
            raw = json.load(f)
        forms = raw if isinstance(raw, list) else raw.get("forms", [])

        page_labels = {}
        for form in forms:
            label = form["document_type"]
            for p in range(form["start_page"], form["end_page"] + 1):
                page_labels[p] = label

        try:
            pages = loader.load(str(pdf_file))
            total_pages += len(pages)
            
            labeled_count = 0
            for page in pages:
                if page.page_number in page_labels:
                    X_text.append(page.text[:max_chars])
                    y_labels.append(page_labels[page.page_number])
                    labeled_count += 1
                elif include_background:
                    X_text.append(page.text[:max_chars])
                    y_labels.append("other")
            
            if labeled_count < len(pages) and not include_background:
                logger.debug(
                    f"{pdf_file.name}: {labeled_count} labeled pages loaded, "
                    f"{len(pages) - labeled_count} pages skipped (no label in target)"
                )
        except Exception as e:
            logger.error(f"Skipping {pdf_file} due to error: {e}")

    if total_pages > len(X_text) and not include_background:
        logger.info(
            f"Training data: {len(X_text)} pages with labels (from {total_pages} total in PDFs; "
            f"{total_pages - len(X_text)} pages unlabeled and skipped)"
        )
    return X_text, y_labels
