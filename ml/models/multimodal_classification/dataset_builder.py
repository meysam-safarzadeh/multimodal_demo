import os
from typing import Any, Callable, Dict, List, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms
from torchvision import transforms as tv_transforms
from models.multimodal_classification.encoders.tabular_utils import TabularNormalizer

class MultiModalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        train_folder: str,
        selected_columns: List[str],
        column_types: Dict[str, str],
        target_column: str,
        tabular_normalizer: Optional[TabularNormalizer] = None,
        transform: Optional[Callable] = None,
        image_loader: Optional[Callable[[str], Any]] = None,
        text_loader: Optional[Callable[[str], str]] = None,
    ):
        self.df = df.copy()
        self.train_folder = train_folder
        self.selected_columns = selected_columns
        self.column_types = column_types
        self.target_column = target_column
        self.tabular_normalizer = tabular_normalizer
        self.df[self.target_column] = self.df[self.target_column].astype(str)
        self.transform = (
            transform
            if transform is not None
            else tv_transforms.Compose([
                tv_transforms.Resize((224, 224)),
                tv_transforms.ToTensor(),
            ])
        )

        # List all files in the train_folder (for fast lookup)
        self.file_map = {os.path.basename(f): os.path.join(dp, f)
                         for dp, dn, filenames in os.walk(train_folder)
                         for f in filenames}

        # Filter rows where all selected columns and target are present and non-empty
        required_columns = selected_columns + [target_column]
        self.df = self.df.dropna(subset=required_columns)
        for col in required_columns:
            self.df = self.df[self.df[col].astype(str).str.strip() != ""]

        self.df.reset_index(drop=True, inplace=True)

        # Create categorical label encoding
        self.label2id = {cat: i for i, cat in enumerate(self.df[target_column].unique())}
        self.id2label = {i: cat for cat, i in self.label2id.items()}

        # Build a category-to-index mapping for each categorical column
        self.cat_maps = {}
        for col, typ in self.column_types.items():
            if typ == "categorical":
                uniq = sorted(self.df[col].unique())
                self.cat_maps[col] = {cat: i for i, cat in enumerate(uniq)}

        self.image_loader = image_loader if image_loader is not None else self.default_image_loader
        self.text_loader = text_loader if text_loader is not None else self.default_text_loader

        self.tabular_cols = [col for col in selected_columns if column_types.get(col) in ("numeric", "categorical")]


    def __len__(self):
        return len(self.df)

    def default_image_loader(self, path):
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img

    def default_text_loader(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def find_file(self, fname):
        # Try to get file by name from the index built in __init__
        return self.file_map.get(os.path.basename(fname), None)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tabular_vals = []
        for col in self.tabular_cols:
            val = row[col]
            tabular_vals.append(val)

        # If you have a normalizer, use it:
        if tabular_vals and self.tabular_normalizer is not None:
            tabular_vals = self.tabular_normalizer.transform_row(row)
        elif tabular_vals:
            # Fallback: encode categoricals, float numerics in order over self.tabular_cols
            for i, col in enumerate(self.tabular_cols):
                col_type = self.column_types.get(col, "unknown")
                if col_type == "categorical":
                    tabular_vals[i] = self.cat_maps[col].get(tabular_vals[i], -1)
                else:
                    tabular_vals[i] = float(tabular_vals[i])

        # Collect non-tabular/modal features
        sample = {}
        for col in self.selected_columns:
            col_type = self.column_types.get(col, "unknown")
            val = row[col]
            if col_type == "image_path":
                file_path = self.find_file(val)
                sample[col] = self.image_loader(file_path) if file_path else None
            elif col_type == "text_path":
                file_path = self.find_file(val)
                sample[col] = self.text_loader(file_path) if file_path else None
            elif col_type == "text":
                sample[col] = val

        if tabular_vals:
            sample["tabular"] = torch.tensor(tabular_vals, dtype=torch.float32)

        label = self.label2id[row[self.target_column]]
        return sample, label


if __name__ == "__main__":
    # --- USAGE EXAMPLE ---
    df = pd.read_csv(r"\datasets\multimodal\cirrhosis_example_file_multimodal.csv")
    train_folder_path = r"\datasets\multimodal\dummy_multiimage_testset"
    selected_columns = ["Age", "Sex", "modality 1", "modality 2"]
    column_types = {'ID': 'numeric', 'N_Days': 'numeric', 'Status': 'categorical', 'Drug': 'categorical', 'Age': 'numeric', 'Sex': 'categorical', 'Ascites': 'categorical', 'Hepatomegaly': 'categorical', 'Spiders': 'categorical', 'Edema': 'categorical', 'Bilirubin': 'numeric', 'Cholesterol': 'numeric', 'Albumin': 'numeric', 'Copper': 'numeric', 'Alk_Phos': 'numeric', 'SGOT': 'numeric', 'Tryglicerides': 'numeric', 'Platelets': 'numeric', 'Prothrombin': 'numeric', 'Stage': 'numeric', 'm1': 'image_path', 'my text': 'text_path', 'm2': 'image_path', 'modality 1': 'image_path', 'modality 2': 'image_path'}
    target_column = "Edema"
    
    dataset = MultiModalDataset(df, train_folder_path, selected_columns, column_types, target_column)

    # Split (e.g., 80% train, 20% val)
    from torch.utils.data import DataLoader, random_split
    import torch
    train_size = int(0.5 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(train_loader))

    print("modality 1 Shape:", batch[0]["modality 1"].shape)
    print("modality 2 Shape:", batch[0]["modality 2"].shape)
    print("Tabular Shape:", batch[0]["tabular"].shape)