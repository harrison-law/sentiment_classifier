"""
The one place that says what the sentiment labels mean.

The models work with the integers 0, 1 and 2 internally. Everything a person reads
(prediction files, tables, plots) uses the words "negative", "neutral" and "positive".
"""

# Integer used inside the model -> word shown to people.
# Saved models were trained with these integers, so do not reorder them.
LABELS = {0: "neutral", 1: "positive", 2: "negative"}

# Word -> integer.
LABEL_IDS = {name: label_id for label_id, name in LABELS.items()}

# Order used in every table and plot.
LABEL_ORDER = ["negative", "neutral", "positive"]


def to_text_labels(values):
    """
    Convert sentiment labels to text. Integers (0, 1, 2), numbers stored as text ("0")
    and words in any letter case ("Negative") are all accepted.

    Args:
        values (iterable): Labels to convert.

    Returns:
        list[str]: One of "negative", "neutral" or "positive" per input value.

    Raises:
        ValueError: If a value is not a known label (this includes missing values).
    """
    lookup = {name: name for name in LABEL_IDS}
    for label_id, name in LABELS.items():
        lookup[str(label_id)] = name
        lookup[f"{label_id}.0"] = name  # pandas turns 0 into 0.0 when a column has gaps

    keys = [str(value).strip().lower() for value in values]
    unknown = sorted({key for key in keys if key not in lookup})
    if unknown:
        raise ValueError(f"Unknown polarity labels {unknown}; expected one of {LABEL_ORDER}")
    return [lookup[key] for key in keys]
