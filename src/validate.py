import json
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict
import argparse
from src.loader import PDFLoader

def load_ground_truth(target_dir: Path) -> Dict[str, Dict[int, str]]:
    """
    Load ground truth labels.
    Returns: {filename: {page_num: label}}
    """
    truth = {}
    for json_file in target_dir.glob("*.json"):
        pdf_name = json_file.stem  # e.g. "dummy1"
        with open(json_file, "r") as f:
            data = json.load(f)
        
        forms = data if isinstance(data, list) else data.get("forms", [])
        page_map = {}
        for form in forms:
            label = form["document_type"]
            for p in range(form["start_page"], form["end_page"] + 1):
                page_map[p] = label
        truth[pdf_name] = page_map
    return truth

def load_predictions(output_dir: Path) -> Dict[str, Dict[int, str]]:
    """
    Load prediction labels.
    Returns: {filename: {page_num: label}}
    """
    preds = {}
    for json_file in output_dir.glob("*.json"):
        pdf_name = json_file.stem
        with open(json_file, "r") as f:
            data = json.load(f)
            
        forms = data if isinstance(data, list) else data.get("forms", [])
        page_map = {}
        for form in forms:
            label = form["document_type"]
            # Filter out "scanned_document" or "unknown" if desired, 
            # but for validation we want to see what the model thinks.
            for p in range(form["start_page"], form["end_page"] + 1):
                page_map[p] = label
        preds[pdf_name] = page_map
    return preds

def get_pdf_page_counts(input_dir: Path) -> Dict[str, int]:
    """
    Get total page count for each PDF to handle unlabeled pages correctly.
    """
    counts = {}
    loader = PDFLoader()
    # We only need to know how many pages exist to check for False Positives on unlabeled pages
    # This might be slow if we parse PDFs. 
    # Alternative: Use the max page number seen in truth/preds? 
    # No, we need total pages to count True Negatives (background correctly identified as background).
    # For speed, let's try to trust the loader or just use max(truth, preds) if speed is an issue.
    # But strictly speaking, we should look at the PDF.
    
    for pdf_file in input_dir.glob("*.pdf"):
        try:
            pages = loader.load(str(pdf_file))
            counts[pdf_file.stem] = len(pages)
        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")
            counts[pdf_file.stem] = 0
    return counts

def calculate_metrics(truth: Dict[str, Dict[int, str]], 
                     preds: Dict[str, Dict[int, str]], 
                     page_counts: Dict[str, int]):
    
    tp = 0  # Correctly identified form
    fp = 0  # Identified form where there is none (or wrong form)
    fn = 0  # Missed a form
    tn = 0  # Correctly identified as no form (background)
    
    misleading_fp = 0 # Specifically: Truth=None, Pred=Form (The "Misleading Mention" case)
    wrong_class_fp = 0 # Truth=FormA, Pred=FormB
    
    for pdf_name, total_pages in page_counts.items():
        pdf_truth = truth.get(pdf_name, {})
        pdf_preds = preds.get(pdf_name, {})
        
        for p in range(1, total_pages + 1):
            true_label = pdf_truth.get(p)
            pred_label = pdf_preds.get(p)
            
            # Normalize labels (e.g. handle "unknown" as None for metric calculation if desired)
            if pred_label in ["unknown", "scanned_document", "manual_review_needed"]:
                pred_label = None
                
            if true_label is not None:
                if pred_label == true_label:
                    tp += 1
                elif pred_label is None:
                    fn += 1
                else:
                    # Predicted a form, but wrong one
                    fp += 1
                    wrong_class_fp += 1
            else:
                # Ground truth is "background"
                if pred_label is None:
                    tn += 1
                else:
                    # Predicted a form where there is none
                    fp += 1
                    misleading_fp += 1
                    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    print("-" * 40)
    print(f"Validation Report")
    print("-" * 40)
    print(f"Total Pages Evaluated: {sum(page_counts.values())}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-" * 40)
    print(f"Detailed Breakdown:")
    print(f"  True Positives (Correct Form): {tp}")
    print(f"  True Negatives (Correct Background): {tn}")
    print(f"  False Negatives (Missed Form): {fn}")
    print(f"  False Positives (Total): {fp}")
    print(f"    - Misleading Mentions (Background -> Form): {misleading_fp}")
    print(f"    - Wrong Class (Form A -> Form B): {wrong_class_fp}")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="data/target", help="Directory containing ground truth JSONs")
    parser.add_argument("--input", default="data/input", help="Directory containing PDFs (for page counts)")
    parser.add_argument("--output", required=True, help="Directory containing prediction JSONs")
    args = parser.parse_args()
    
    truth_labels = load_ground_truth(Path(args.target))
    pred_labels = load_predictions(Path(args.output))
    page_counts = get_pdf_page_counts(Path(args.input))
    
    calculate_metrics(truth_labels, pred_labels, page_counts)
