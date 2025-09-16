import pandas as pd
import re
from typing import List, Dict, Any, Optional


class MetadataDetector:
    """
    Utility class for detecting modalities, dataset characteristics, and possible targets
    from user-uploaded CSV metadata.
    """

    FILE_PATTERN = re.compile(
        r'.*\.(jpg|jpeg|png|bmp|gif|tiff|wav|mp3|flac|aac|csv|txt|xlsx|xls|pdf)$', re.I
    )
    IMAGE_PATTERN = re.compile(
        r'\.(?:jpg|jpeg|png|bmp|gif|tiff)$', re.I
    )
    AUDIO_PATTERN = re.compile(
        r'\.(?:wav|mp3|flac|aac)$', re.I
    )
    TEXTFILE_PATTERN = re.compile(
        r'\.(?:txt|md|rtf|html)$', re.I
    )

    def __init__(self, csv_path: str, max_rows: int = 10000):
        """
        Args:
            csv_path (str): Path to the CSV file.
            max_rows (int): Max number of rows to load for profiling.
        """
        self.csv_path = csv_path
        self.max_rows = max_rows
        self.df = self._load_and_clean_csv()

    def _load_and_clean_csv(self) -> pd.DataFrame:
        """Loads the CSV and cleans string columns, skipping file paths."""
        df = pd.read_csv(self.csv_path, nrows=self.max_rows)
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].apply(self.clean_text_skip_files)
        return df

    @staticmethod
    def get_clean_sample_vals(series: pd.Series, n: int = 10) -> pd.Series:
        """Returns a sample of up to n non-null, non-'nan' values as strings."""
        vals = series.dropna().astype(str)
        vals = vals[~vals.str.strip().str.lower().eq('nan')]
        return vals.sample(min(n, len(vals)), random_state=42) if len(vals) > 0 else pd.Series([], dtype=str)

    @classmethod
    def clean_text_skip_files(cls, val: Any) -> Any:
        """Lowercases and strips non-file-path strings; leaves file paths unchanged."""
        if pd.isnull(val):
            return val
        val_str = str(val).strip()
        if cls.FILE_PATTERN.match(val_str):
            return val_str
        return val_str.lower()

    def detect(self) -> Dict[str, Any]:
        """Detects modalities, dataset stats, candidate targets, and column types."""
        modalities = set()
        column_types = {}
        target_columns_categorical: List[str] = []
        feature_columns: List[str] = []
        modality_columns: List[str] = []
        other_columns: List[str] = []

        for col in self.df.columns:
            series = self.df[col]
            sample_vals = self.get_clean_sample_vals(series)
            n_unique = series.nunique(dropna=True)
            avg_str_len = sample_vals.str.len().mean() if not sample_vals.empty else 0

            if sample_vals.str.contains(self.IMAGE_PATTERN).any():
                column_types[col] = 'image_path'
                modality_columns.append(col)
                modalities.add('image')
            elif sample_vals.str.contains(self.AUDIO_PATTERN).any():
                column_types[col] = 'audio_path'
                modality_columns.append(col)
                modalities.add('audio')
            elif sample_vals.str.contains(self.TEXTFILE_PATTERN).any():
                column_types[col] = 'text_path'
                modality_columns.append(col)
                modalities.add('text')
            elif avg_str_len > 20 and n_unique > 10:
                column_types[col] = 'text'
                feature_columns.append(col)
                modalities.add('text')
            elif (pd.api.types.is_numeric_dtype(series) and n_unique == len(series)):
                column_types[col] = 'other'
                other_columns.append(col)
            elif n_unique < 20:
                column_types[col] = 'categorical'
                target_columns_categorical.append(col)
                feature_columns.append(col)
                modalities.add('tabular')
            elif pd.api.types.is_numeric_dtype(series):
                column_types[col] = 'numeric'
                feature_columns.append(col)
                modalities.add('tabular')
            else:
                column_types[col] = 'other'
                other_columns.append(col)

        n_rows, n_columns = self.df.shape
        n_classes_dict = {col: self.df[col].nunique(dropna=True) for col in target_columns_categorical}
        missing_data = self.df.isnull().sum().to_dict()

        return {
            'modalities': sorted(list(modalities)),
            'n_rows': n_rows,
            'n_columns': n_columns,
            'feature_columns': feature_columns,
            'modality_columns': modality_columns,
            'target_columns_categorical': target_columns_categorical,
            'other_columns': other_columns,
            'n_classes': n_classes_dict,
            'column_types': column_types,
            'missing_data': missing_data,
        }


if __name__ == "__main__":
    detector = MetadataDetector("/home/meysam/multimodal_demo_files/multimodal/dummy_multiimage_testset")
    summary = detector.detect()
    print(summary)
