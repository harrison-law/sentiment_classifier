"""
Find and load the models that can be evaluated.

Any Hugging Face model that works with AutoTokenizer and AutoModelForSequenceClassification
can be used by name, once it has been trained with exec/train_model.py. No code changes are
needed for those.

REGISTRY is only for other kinds of models, e.g. a SentimentModel subclass written by hand.
"""
import json
from pathlib import Path

# Trained models are saved as <models_dir>/github-novielli-<model name>
MODEL_FILE_PREFIX = "github-novielli-"


def default_model_path(model_name, models_dir):
    """
    Where exec/train_model.py saves a model by default, and where evaluation looks for it.

    Args:
        model_name (str): Hugging Face name, e.g. "bert-base-cased" or "cardiffnlp/twitter-roberta-base".
        models_dir (str or Path): Folder that holds the trained models.

    Returns:
        Path: e.g. <models_dir>/github-novielli-bert-base-cased. A "/" in the name becomes "__".
    """
    return Path(models_dir) / f"{MODEL_FILE_PREFIX}{model_name.replace('/', '__')}"


class SentimentModel:
    """What every model must be able to do. Write a subclass to add a new kind of model."""

    def load(self):
        """Load the model into memory. Called once, before predict."""

    def predict(self, texts):
        """
        Args:
            texts (list[str]): Texts to classify.

        Returns:
            list[str]: "negative", "neutral" or "positive" for each text, in the same order.
        """
        raise NotImplementedError


class TransformerModel(SentimentModel):
    """
    A Hugging Face model trained with exec/train_model.py.

    Args:
        model_name (str): Hugging Face name, e.g. "bert-base-cased". This must be the
            --model_name the model was trained with.
        saved_path (str or Path): The file written by exec/train_model.py (its --output).
    """

    def __init__(self, model_name, saved_path):
        self.model_name = model_name
        self.saved_path = Path(saved_path)

    def load(self):
        if not self.saved_path.exists():
            raise FileNotFoundError(
                f"{self.model_name} has not been trained yet ({self.saved_path} not found). Train it with:\n"
                f"  python exec/train_model.py --splits_dir <splits folder> "
                f"--model_name {self.model_name} --output {self.saved_path}"
            )
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from api.model import load_model

        try:
            self.model, self.tokenizer = load_model(
                str(self.saved_path), AutoModelForSequenceClassification, AutoTokenizer, self.model_name
            )
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(
                f'Could not load "{self.model_name}" from Hugging Face. The name must be a Hugging Face '
                f"model that works with AutoTokenizer and AutoModelForSequenceClassification, and it must "
                f"be the same --model_name that {self.saved_path} was trained with."
            ) from error
        except RuntimeError as error:
            raise ValueError(
                f"{self.saved_path} does not match {self.model_name}. Check that it was trained with "
                f"--model_name {self.model_name}."
            ) from error

    def predict(self, texts):
        from api.model import predict_labels
        return predict_labels(self.model, self.tokenizer, texts)


# Name -> function that receives the models folder and returns a SentimentModel.
# Hugging Face models do not need an entry: any name not listed here is loaded as one.
# Example of a model that is not a transformer:
#   "always-neutral": lambda models_dir: AlwaysNeutral(),
REGISTRY = {}


def load_registered_model(name, models_dir, saved_path=None):
    """
    Load a model by name.

    Args:
        name (str): A name in REGISTRY, or any Hugging Face model name trained with exec/train_model.py.
        models_dir (str or Path): Folder that holds the trained models.
        saved_path (str or Path, optional): The trained model file of a Hugging Face model.
            Defaults to default_model_path(name, models_dir).

    Returns:
        SentimentModel: The model, ready for predict.
    """
    if name in REGISTRY:
        model = REGISTRY[name](models_dir)
    else:
        model = TransformerModel(name, saved_path or default_model_path(name, models_dir))
    model.load()
    return model


def list_trained_models(models_dir):
    """
    Find the Hugging Face models that have been trained into models_dir.

    Args:
        models_dir (str or Path): Folder that holds the trained models.

    Returns:
        list[str]: Model names that can be used with load_registered_model, sorted.
    """
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []

    names = set()
    for path in models_dir.glob(f"{MODEL_FILE_PREFIX}*"):
        if not path.is_file() or path.suffix == ".json":
            continue
        name = path.name[len(MODEL_FILE_PREFIX):].replace("__", "/")
        # The metrics file written by training holds the exact name, so prefer it.
        metrics_path = path.with_name(path.stem + "_metrics.json")
        if metrics_path.exists():
            try:
                name = json.loads(metrics_path.read_text()).get("model_name", name)
            except (OSError, ValueError):
                pass
        names.add(name)
    return sorted(names)
