"""
Evaluate sentiment models. Both notebooks use these functions, so every model is
measured in exactly the same way.

Terms used below:
    y_true: the correct labels, as text.
    y_pred: the labels the model predicted, as text.
    split:  which part of the data was used: "train", "val" or "test".
"""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from api.labels import LABEL_ORDER, to_text_labels

SPLITS = ["train", "val", "test"]
SPLIT_TITLES = {"train": "Training", "val": "Validation", "test": "Test"}

# One blue, light to dark, so a darker cell always means a higher percentage.
PERCENT_COLORS = LinearSegmentedColormap.from_list(
    "percent_blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def split_path(splits_dir, split):
    """Return the path of a split file, e.g. <splits_dir>/val.csv."""
    return Path(splits_dir) / f"{split}.csv"


def load_split(splits_dir, split):
    """
    Load train.csv, val.csv or test.csv and make sure its labels are text.

    Args:
        splits_dir (str or Path): Folder created by exec/make_splits.py.
        split (str): "train", "val" or "test".

    Returns:
        pd.DataFrame: Data with at least 'text' and 'polarity' columns.
    """
    path = split_path(splits_dir, split)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Create the split files with exec/make_splits.py first.")
    df = pd.read_csv(path)
    df["polarity"] = to_text_labels(df["polarity"])
    return df


def predict(model, texts):
    """
    Run a model on some texts.

    Args:
        model (SentimentModel): A loaded model (see api/model_registry.py).
        texts (iterable): The texts to classify.

    Returns:
        list[str]: "negative", "neutral" or "positive" for each text.
    """
    texts = list(texts)
    labels = to_text_labels(model.predict(texts))
    if len(labels) != len(texts):
        raise ValueError(f"The model returned {len(labels)} labels for {len(texts)} texts.")
    return labels


def confusion_matrix(y_true, y_pred):
    """
    Count how often each true label received each predicted label.

    Returns:
        pd.DataFrame: Counts, with true labels as rows and predicted labels as columns.
    """
    matrix = sklearn_confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    return pd.DataFrame(
        matrix,
        index=pd.Index(LABEL_ORDER, name="True label"),
        columns=pd.Index(LABEL_ORDER, name="Predicted label"),
    )


def _row_percents(counts):
    """Turn counts into percentages of each row. A row with no texts stays all zeros."""
    row_totals = counts.sum(axis=1)
    return counts.div(row_totals.where(row_totals > 0, 1), axis=0) * 100


def plot_confusion_matrix(counts, model_name, split, ax=None, save_path=None):
    """
    Draw a confusion matrix. Color shows the percentage of each true label (always on a
    0-100% scale, so plots can be compared). Each cell shows the percentage and the count.

    Args:
        counts (pd.DataFrame): Counts returned by confusion_matrix.
        model_name (str): Shown in the title.
        split (str): "train", "val" or "test". Shown in the title.
        ax (matplotlib axes, optional): Draw here instead of in a new figure.
        save_path (str or Path, optional): Also save the figure as a PNG.

    Returns:
        matplotlib axes: The axes that were drawn on.
    """
    percents = _row_percents(counts)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5), layout="constrained")

    image = ax.imshow(percents.values, cmap=PERCENT_COLORS, vmin=0, vmax=100)
    for row in range(len(LABEL_ORDER)):
        for col in range(len(LABEL_ORDER)):
            percent = percents.iloc[row, col]
            ax.text(
                col, row, f"{percent:.1f}%\n({counts.iloc[row, col]})",
                ha="center", va="center", fontsize=11,
                color="white" if percent >= 50 else "#0b0b0b",
            )

    ax.set_xticks(range(len(LABEL_ORDER)), LABEL_ORDER)
    ax.set_yticks(range(len(LABEL_ORDER)), LABEL_ORDER)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{model_name}\n{SPLIT_TITLES.get(split, split)} set ({int(counts.values.sum())} texts)")

    # White gaps between cells
    ax.set_xticks([0.5, 1.5], minor=True)
    ax.set_yticks([0.5, 1.5], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    ax.spines[:].set_visible(False)

    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("% of true label")
    colorbar.outline.set_visible(False)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    return ax


def compute_metrics(y_true, y_pred):
    """
    Compute summary scores. All scores go from 0 (worst) to 1 (best).

    Returns:
        dict: n (number of texts), accuracy (share predicted correctly), macro_f1 (F1
        averaged so that each label counts equally), and per_class, with precision,
        recall, f1 and support (number of texts) for each label.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_ORDER, zero_division=0
    )
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABEL_ORDER)
        },
    }


def flatten_metrics(metrics):
    """Turn the result of compute_metrics into one flat row (a dict) for tables."""
    row = {key: metrics[key] for key in ["n", "accuracy", "macro_f1"]}
    for label in LABEL_ORDER:
        for score in ["precision", "recall", "f1"]:
            row[f"{score}_{label}"] = metrics["per_class"][label][score]
    return row


def flatten_confusion_matrix(counts, model_name, split):
    """
    Put a confusion matrix on one row, with a count and a percentage (of the true label)
    column per true:predicted pair. For example "negative:neutral_count" is how many
    negative texts were predicted neutral.

    Args:
        counts (pd.DataFrame): Counts returned by confusion_matrix.
        model_name (str): Stored in the 'model' column.
        split (str): Stored in the 'split' column.

    Returns:
        pd.DataFrame: One row.
    """
    tables = {"count": counts, "percent": _row_percents(counts)}

    row = {"model": model_name, "split": split}
    for kind, table in tables.items():
        for true_label in LABEL_ORDER:
            for predicted_label in LABEL_ORDER:
                row[f"{true_label}:{predicted_label}_{kind}"] = table.loc[true_label, predicted_label]
    return pd.DataFrame([row])


def unflatten_confusion_matrix(row):
    """Rebuild the counts table from one row of load_all_results."""
    return pd.DataFrame(
        [[int(row[f"{true}:{predicted}_count"]) for predicted in LABEL_ORDER] for true in LABEL_ORDER],
        index=pd.Index(LABEL_ORDER, name="True label"),
        columns=pd.Index(LABEL_ORDER, name="Predicted label"),
    )


def save_results(model_name, split, y_true, y_pred, results_dir, data_path=None):
    """
    Save everything the comparison notebook needs to <results_dir>/<model_name>/:
        <split>_confusion_matrix.png, <split>_metrics.json, <split>_confusion_flat.csv

    Args:
        model_name (str): Name of the model.
        split (str): "train", "val" or "test".
        y_true (list[str]): Correct labels.
        y_pred (list[str]): Predicted labels.
        results_dir (str or Path): Results folder.
        data_path (str or Path, optional): The split file that was used. A short fingerprint
            of it is saved, to detect models that were evaluated on different data.

    Returns:
        Path: The folder that was written to.
    """
    folder = Path(results_dir) / model_name.replace("/", "__")
    folder.mkdir(parents=True, exist_ok=True)

    counts = confusion_matrix(y_true, y_pred)
    ax = plot_confusion_matrix(counts, model_name, split, save_path=folder / f"{split}_confusion_matrix.png")
    plt.close(ax.figure)

    flatten_confusion_matrix(counts, model_name, split).to_csv(folder / f"{split}_confusion_flat.csv", index=False)

    metrics = {"model": model_name, "split": split, **compute_metrics(y_true, y_pred)}
    if data_path is not None:
        metrics["data_fingerprint"] = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()[:16]
    (folder / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return folder


def load_all_results(results_dir):
    """
    Find and load the saved results of every evaluated model.

    Returns:
        pd.DataFrame: One row per model and split, with the metrics and the flattened
        confusion matrix, sorted by model and then train, val, test.
    """
    rows = []
    for metrics_path in sorted(Path(results_dir).glob("*/*_metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        flat_path = metrics_path.with_name(metrics_path.name.replace("_metrics.json", "_confusion_flat.csv"))
        flat = pd.read_csv(flat_path).iloc[0].to_dict()
        rows.append({
            **flat,  # already contains model and split
            **flatten_metrics(metrics),
            "data_fingerprint": metrics.get("data_fingerprint"),
        })

    if not rows:
        raise FileNotFoundError(f"No results in {results_dir}. Run notebooks/evaluate_model.ipynb first.")

    results = pd.DataFrame(rows)
    results["split"] = pd.Categorical(results["split"], SPLITS, ordered=True)
    return results.sort_values(["model", "split"]).reset_index(drop=True)


def pick_best_model(results):
    """
    Choose the model with the highest validation macro F1. If models tie, validation
    accuracy decides, then alphabetical order. Test results are never used.

    Args:
        results (pd.DataFrame): Table returned by load_all_results.

    Returns:
        str: Name of the best model.
    """
    val = results[results["split"] == "val"]
    if val.empty:
        raise ValueError("No model has validation results yet.")
    ranked = val.sort_values(["macro_f1", "accuracy", "model"], ascending=[False, False, True])
    return ranked.iloc[0]["model"]
