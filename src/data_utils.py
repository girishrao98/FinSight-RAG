import os
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_csv_dataset(path: str, text_column: str = "text") -> pd.DataFrame:
    """Load a CSV dataset and return the text column."""
    df = pd.read_csv(path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {path}")
    return df[[text_column]].rename(columns={text_column: "text"})


def normalize_text(text: str) -> str:
    """Normalize raw text for indexing."""
    return " ".join(text.strip().split())


def build_embeddings(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Build sentence embeddings for a list of texts."""
    model = SentenceTransformer(model_name)
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)


def save_embeddings(path: str, embeddings: np.ndarray) -> None:
    """Save embeddings to disk as a NumPy file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embeddings)


def load_embeddings(path: str) -> np.ndarray:
    """Load embeddings from a NumPy file."""
    return np.load(path)
