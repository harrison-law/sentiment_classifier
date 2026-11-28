"""
Split a labeled dataset into train.csv, val.csv and test.csv, once, with text labels.

Every model is trained and judged on these same files, which keeps comparisons fair.

Example:
    python exec/make_splits.py --input <labeled.csv> --output_dir <splits folder>
"""
import sys
import os
import argparse
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from api.labels import LABEL_ORDER, to_text_labels
from api.split_dataset import split_dataset

COLUMNS = ["comment_id", "text", "polarity", "updated_at"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Labeled CSV with text, polarity and updated_at columns")
    parser.add_argument("--output_dir", required=True, help="Folder for train.csv, val.csv and test.csv")
    parser.add_argument("--force", action="store_true", help="Replace existing split files")
    return parser.parse_args()

def main():
    args = parse_args()

    paths = {name: os.path.join(args.output_dir, f"{name}.csv") for name in ["train", "val", "test"]}
    existing = [path for path in paths.values() if os.path.exists(path)]
    if existing and not args.force:
        sys.exit(f"Split files already exist: {existing}\nUse --force to replace them.")

    df = pd.read_csv(args.input)
    df["polarity"] = to_text_labels(df["polarity"])

    # Same settings as exec/train_model.py, so the rows match models trained with --input.
    splits = dict(zip(paths, split_dataset(df, val_frac=0.1, test_frac=0.2, gap=pd.Timedelta(0))))

    os.makedirs(args.output_dir, exist_ok=True)
    for name, split_df in splits.items():
        columns = [column for column in COLUMNS if column in split_df.columns]
        split_df[columns].to_csv(paths[name], index=False)
        counts = split_df["polarity"].value_counts().reindex(LABEL_ORDER, fill_value=0).to_dict()
        print(f"{name}: {len(split_df)} rows {counts} -> {paths[name]}")

if __name__ == "__main__":
    main()
