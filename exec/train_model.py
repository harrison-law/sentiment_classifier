import sys
import json
import os
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from api.model import train_model, evaluate_model
from api.split_dataset import split_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--metrics_output",
        default=None,
        help="Path for the JSON metrics file. Defaults to <output without extension>_metrics.json",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    
    train_df, val_df, test_df = split_dataset(
        df=df, 
        val_frac=0.1, 
        test_frac=0.2, 
        gap=pd.Timedelta(0)
    )

    print(f"Dataset split completed:")
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    history = []
    saved_model_path = train_model(
        train_df=train_df,
        val_df=val_df,
        model_save_path=args.output,
        model_task=AutoModelForSequenceClassification,
        model_tokenizer=AutoTokenizer,
        model_name=args.model_name,
        history=history,
    )

    test_metrics = evaluate_model(
        test_df=test_df,
        model_saved_path=saved_model_path,
        model_task=AutoModelForSequenceClassification,
        model_tokenizer=AutoTokenizer,
        model_name=args.model_name,
    )

    metrics_path = args.metrics_output or os.path.splitext(args.output)[0] + "_metrics.json"
    metrics = {
        "model_name": args.model_name,
        "split_sizes": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "history": history,
        "test": test_metrics,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"MODEL_SAVED_AT: {saved_model_path}")
    print(f"METRICS_SAVED_AT: {metrics_path}")

if __name__ == "__main__":
    main()