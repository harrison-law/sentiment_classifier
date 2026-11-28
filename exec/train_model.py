import sys
import json
import os
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from api.model import train_model, evaluate_model
from api.split_dataset import split_dataset
from api.model_registry import default_model_path

DEFAULT_MODELS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "rawdata", "kaiaulu", "sentiment", "sentiment_classifier", "model"
)

def parse_args():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="One labeled CSV, split into train/validation/test here")
    source.add_argument("--splits_dir", help="Folder with train.csv, val.csv and test.csv (see exec/make_splits.py)")
    parser.add_argument(
        "--model_name",
        required=True,
        help="Any Hugging Face model that works with AutoModelForSequenceClassification, e.g. bert-base-cased",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to save the trained model. Defaults to <models_dir>/github-novielli-<model_name>, "
             "where the evaluation notebook looks for it",
    )
    parser.add_argument(
        "--models_dir",
        default=DEFAULT_MODELS_DIR,
        help="Folder for trained models, used when --output is not given",
    )
    parser.add_argument(
        "--metrics_output",
        default=None,
        help="Path for the JSON metrics file. Defaults to <output without extension>_metrics.json",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    if args.output is None:
        args.output = str(default_model_path(args.model_name, os.path.normpath(args.models_dir)))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if args.splits_dir:
        train_df, val_df, test_df = [
            pd.read_csv(os.path.join(args.splits_dir, f"{name}.csv")) for name in ["train", "val", "test"]
        ]
    else:
        train_df, val_df, test_df = split_dataset(
            df=pd.read_csv(args.input),
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