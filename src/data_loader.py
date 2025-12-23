import os
import urllib.request
from typing import Tuple, Optional

import pandas as pd
import numpy as np
import sklearn


def load_parkinsons_csv(path: str, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a Parkinson's dataset from a CSV file and return features and target.

    Behavior:
    - If `target_column` is provided and exists in the CSV, it will be used.
    - Otherwise, tries common target names: 'status', 'target', 'label'.
    - If none found, falls back to using the last column as the target.

    Returns:
    - X: `pd.DataFrame` of features (target column dropped)
    - y: `pd.Series` of target values
    """
    df = pd.read_csv(path)

    # Determine target column
    if target_column and target_column in df.columns:
        target_col = target_column
    else:
        for cand in ("status", "target", "label"):
            if cand in df.columns:
                target_col = cand
                break
        else:
            target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def download_parkinsons_from_uci(dest_path: Optional[str] = None,
                                  url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data") -> str:
    """Download the Parkinson's dataset from the UCI repo and save to `dest_path`.

    If `dest_path` is None, saves to the repository `data/parkinsons.csv` file.
    Returns the absolute path to the saved file.
    """
    if dest_path is None:
        repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
        dest_path = os.path.join(repo_root, "data", "parkinsons.csv")

    dest_path = os.path.abspath(dest_path)
    dest_dir = os.path.dirname(dest_path) or "."
    os.makedirs(dest_dir, exist_ok=True)

    with urllib.request.urlopen(url) as resp:
        content = resp.read()

    with open(dest_path, "wb") as f:
        f.write(content)

    return dest_path


def display_dataset_info(X: pd.DataFrame, y: Optional[pd.Series] = None, target_column: Optional[str] = None) -> dict:
    """Display basic dataset information and return a summary dict.

    Printed information:
    - Shape
    - Columns
    - Data types
    - Missing value counts and percentages
    - Target distribution (if `y` provided or `target_column` present)

    Returns a dictionary with the same information for programmatic use.
    """
    # If target column is provided and exists in X, extract it
    if target_column and target_column in X.columns:
        y = X[target_column]
        X = X.drop(columns=[target_column])

    n_rows = len(X)
    info = {}
    info["shape"] = X.shape
    info["columns"] = list(X.columns)
    info["dtypes"] = X.dtypes.astype(str).to_dict()

    missing_counts = X.isna().sum()
    missing_pct = (missing_counts / n_rows * 100).round(3)
    missing_df = pd.DataFrame({"count": missing_counts, "pct": missing_pct})
    info["missing"] = missing_df

    # Print summary
    print("Dataset shape:", X.shape)
    print("Columns:", ", ".join(X.columns))
    print("\nData types:")
    print(X.dtypes)

    if missing_counts.sum() > 0:
        print("\nMissing values (count, %):")
        print(missing_df[missing_df["count"] > 0])
    else:
        print("\nNo missing values detected.")

    # Target distribution
    if y is not None:
        tgt_counts = y.value_counts(dropna=False)
        tgt_pct = (tgt_counts / len(y) * 100).round(3)
        tgt_df = pd.DataFrame({"count": tgt_counts, "pct": tgt_pct})
        info["target_distribution"] = tgt_df
        print("\nTarget distribution:")
        print(tgt_df)

    return info
import os
import urllib.request
import pandas as pd
import numpy as np
import sklearn
from typing import Tuple, Optional


def load_parkinsons_csv(path: str, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a Parkinson's dataset from a CSV file and return features and target.

    Behavior:
    - If `target_column` is provided and exists in the CSV, it will be used.
    - Otherwise, tries common target names: 'status', 'target', 'label'.
    - If none found, falls back to using the last column as the target.

    Returns:
    - X: `pd.DataFrame` of features (target column dropped)
    - y: `pd.Series` of target values
    """
    df = pd.read_csv(path)

    # Determine target column
    if target_column and target_column in df.columns:
        target_col = target_column
    else:
        for cand in ("status", "target", "label"):
            if cand in df.columns:
                target_col = cand
                break
        else:
            target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def download_parkinsons_from_uci(dest_path: Optional[str] = None,
                                  url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data") -> str:
    """Download the Parkinson's dataset from the UCI repo and save to `dest_path`.

    If `dest_path` is None, saves to the repository `data/parkinsons.csv` file.
    Returns the absolute path to the saved file.
    """
    if dest_path is None:
        repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
        dest_path = os.path.join(repo_root, "data", "parkinsons.csv")

    dest_path = os.path.abspath(dest_path)
    dest_dir = os.path.dirname(dest_path) or "."
    os.makedirs(dest_dir, exist_ok=True)

    with urllib.request.urlopen(url) as resp:
        content = resp.read()

    with open(dest_path, "wb") as f:
        f.write(content)

    return dest_path
import pandas as pd
import numpy as np
import sklearn
from typing import Tuple, Optional


def load_parkinsons_csv(path: str, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
	"""Load a Parkinson's dataset from a CSV file and return features and target.

	Behavior:
	- If `target_column` is provided and exists in the CSV, it will be used.
	- Otherwise, tries common target names: 'status', 'target', 'label'.
	- If none found, falls back to using the last column as the target.

	Returns:
	- X: `pd.DataFrame` of features (target column dropped)
	- y: `pd.Series` of target values
	"""
	df = pd.read_csv(path)

	# Determine target column
	if target_column and target_column in df.columns:
		target_col = target_column
	else:
		for cand in ("status", "target", "label"):
			if cand in df.columns:
				target_col = cand
				break
		else:
			target_col = df.columns[-1]

	X = df.drop(columns=[target_col])
	y = df[target_col]
	return X, y


