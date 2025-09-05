import numpy as np
import pandas as pd
from typing import Dict, List, Any


class TabularNormalizer:
    """
    Standardizes numeric columns and encodes categorical columns using pre-fit statistics and maps.
    """

    def __init__(self):
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.numeric_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.cat_maps: Dict[str, Dict[Any, int]] = {}
        self.fitted: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        column_types: Dict[str, str],
        cat_maps: Dict[str, Dict[Any, int]],
        selected_columns: List[str],
    ):
        self.numeric_cols = [col for col in selected_columns if column_types.get(col) == "numeric"]
        self.cat_cols = [col for col in selected_columns if column_types.get(col) == "categorical"]
        self.cat_maps = cat_maps

        for col in self.numeric_cols:
            col_vals = df[col].astype(float)
            self.means[col] = col_vals.mean()
            self.stds[col] = col_vals.std() if col_vals.std() > 0 else 1.0

        self.fitted = True

    def transform_row(self, row: pd.Series) -> List[float]:
        """
        Transforms a single row: normalizes numeric columns and encodes categoricals.
        """
        assert self.fitted, "TabularNormalizer not fitted yet!"
        normed = []
        for col in self.numeric_cols:
            val = float(row[col])
            normed.append((val - self.means[col]) / self.stds[col])
        for col in self.cat_cols:
            val = row[col]
            normed.append(self.cat_maps[col].get(val, -1))  # -1 for unknown
        return normed

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms an entire DataFrame.
        Only normalizes/encodes numeric and categorical columns in-place, keeps the rest untouched.
        Returns a new DataFrame.
        """
        assert self.fitted, "TabularNormalizer not fitted yet!"
        out = df.copy()
        # Normalize numeric columns
        for col in self.numeric_cols:
            out[col] = (df[col].astype(float) - self.means[col]) / self.stds[col]
        # Encode categorical columns
        for col in self.cat_cols:
            out[col] = df[col].map(self.cat_maps[col]).fillna(-1).astype(int)
        return out

    def save(self, path):
        np.savez(path,
                 means=self.means,
                 stds=self.stds,
                 numeric_cols=self.numeric_cols,
                 cat_cols=self.cat_cols,
                 cat_maps=self.cat_maps)

    @classmethod
    def load(cls, path):
        arr = np.load(path, allow_pickle=True)
        obj = cls()
        obj.means = arr["means"].item()
        obj.stds = arr["stds"].item()
        obj.numeric_cols = list(arr["numeric_cols"])
        obj.cat_cols = list(arr["cat_cols"])
        obj.cat_maps = arr["cat_maps"].item()
        obj.fitted = True
        return obj
