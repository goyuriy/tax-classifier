import argparse
import pickle
from pathlib import Path
from src.models import HybridClassifier, SemanticClassifier
from src.pipeline import DocumentClassifierPipeline
from src.train_utils import load_training_data
from src.utils import setup_logger

logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Tax Document Classifier")
    parser.add_argument("--mode", choices=["train", "predict"], default="predict")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--model_path", default="models/semantic_classifier.pkl")
    parser.add_argument("--output_dir", default="data/output")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        logger.info("Loading training data...")
        X, y = load_training_data(args.data_dir, max_chars=50)
        
        if not X:
            logger.error("No training data found or loaded.")
            return

        logger.info(f"Loaded {len(X)} samples.")
        
        clf = SemanticClassifier()
        clf.fit(X, y)
        
        # Save
        Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
        clf.save(args.model_path)
        logger.info(f"Model saved to {args.model_path}")
        
    elif args.mode == "predict":
        # Load model if exists, else use heuristics only (or empty semantic)
        semantic = None
        try:
            semantic = SemanticClassifier.load(args.model_path)
            logger.info("Loaded Semantic Classifier.")
        except FileNotFoundError:
            logger.warning(f"Semantic Classifier model not found at {args.model_path}. Using Heuristics only (or untrained semantic fallback).")
            semantic = SemanticClassifier() # Untrained
            
        hybrid = HybridClassifier(semantic_classifier=semantic)
        pipeline = DocumentClassifierPipeline(hybrid)
        
        pipeline.process_directory(f"{args.data_dir}/input", args.output_dir)

if __name__ == "__main__":
    main()
