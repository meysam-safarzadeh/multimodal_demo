from typing import List, Dict, Optional, Tuple, Any


def build_multimodal_rgb_config(
    modalities,
    num_classes=10,
    image_size=(224, 224),
    backbone_map=None,
    out_dim_map=None,
    in_channels_map=None,
    fusion_type="concat",
    fusion_out_dim=None,
    use_norm=True,
    head_type="classifier",
    hidden_dims=[512, 128],
    dropout=0.2,
    activation="gelu",
    use_batchnorm=True,
):
    """
    Args:
        modalities: list of str, e.g. ['rgb_01', 'thermal', 'rgb_02']
        num_classes: int
        image_size: tuple
        backbone_map: dict mapping modality -> backbone (optional)
        out_dim_map: dict mapping modality -> out_dim (optional)
        in_channels_map: dict mapping modality -> in_channels (optional)
        fusion_type: str, default "concat"
        fusion_out_dim: int, if None, inferred as sum of out_dims
        use_norm: bool
        head_type: str, default "classifier"
        hidden_dims: list of int
        dropout: float
        activation: str
        use_batchnorm: bool
    Returns:
        config: dict
    """
    # Sensible general defaults (used if not specified for a modality)
    default_backbone = "resnet18"
    default_out_dim = 128
    default_in_channels = 3  # change to 1 if most of your modalities are 1-channel

    backbone_map = backbone_map or {}
    out_dim_map = out_dim_map or {}
    in_channels_map = in_channels_map or {}

    modal_conf = {}
    out_dims = []
    for m in modalities:
        modal_conf[m] = {
            "type": "image",
            "backbone": backbone_map.get(m, default_backbone),
            "out_dim": out_dim_map.get(m, default_out_dim),
            "in_channels": in_channels_map.get(m, default_in_channels),
        }
        out_dims.append(modal_conf[m]["out_dim"])

    # Fusion out_dim (sum for concat, or user override)
    if fusion_out_dim is None:
        if fusion_type == "concat":
            fusion_out_dim = sum(out_dims)
        else:
            fusion_out_dim = max(out_dims)  # or choose a sensible default

    config = {
        "modalities": modal_conf,
        "fusion": {
            "type": fusion_type,
            "out_dim": fusion_out_dim,
            "use_norm": use_norm,
        },
        "head": {
            "type": head_type,
            "num_classes": num_classes,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "activation": activation,
            "use_batchnorm": use_batchnorm,
            "image_sizes": [image_size] * len(modalities),
        },
    }
    return config


def build_universal_multimodal_config(
    selected_columns: List[str],
    column_types: Dict[str, str],
    target_column: str,
    num_classes: int,
    image_size: Tuple[int, int] = (224, 224),
    backbone_map: Optional[Dict[str, str]] = None,
    out_dim_map: Optional[Dict[str, int]] = None,
    in_channels_map: Optional[Dict[str, int]] = None,
    fusion_type: str = "concat",
    fusion_out_dim: Optional[int] = None,
    use_norm: bool = True,
    head_type: str = "classifier",
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.2,
    activation: str = "gelu",
    use_batchnorm: bool = True,
) -> Dict[str, Any]:
    """
    Build a universal multimodal model config for image, tabular, and text features.

    Args:
        selected_columns: Feature columns to use.
        column_types: Mapping from column name to type ('image_path', 'text', etc.).
        target_column: Name of the target column.
        num_classes: Number of output classes.
        image_size: Image input size.
        backbone_map: Optional backbone override per column.
        out_dim_map: Optional output dimension override per column.
        in_channels_map: Optional input channel override per column.
        fusion_type: Fusion strategy.
        fusion_out_dim: Output dim after fusion (None = infer).
        use_norm: Whether to use normalization in fusion.
        head_type: Type of prediction head.
        hidden_dims: List of hidden dims for classifier head.
        dropout: Dropout probability for head.
        activation: Activation for head.
        use_batchnorm: BatchNorm in head.

    Returns:
        A configuration dictionary for multimodal model building.
    """
    # Defaults
    hidden_dims = hidden_dims or [512, 128]
    backbone_map = backbone_map or {}
    out_dim_map = out_dim_map or {}
    in_channels_map = in_channels_map or {}

    modalities_conf = {}
    out_dims = []

    # Group columns by modality
    tabular_columns = []

    for col in selected_columns:
        col_type = column_types.get(col, "unknown")
        if col_type == "image_path":
            # Each image column is its own modality
            modalities_conf[col] = {
                "type": "image",
                "backbone": backbone_map.get(col, "resnet18"),
                "out_dim": out_dim_map.get(col, 128),
                "in_channels": in_channels_map.get(col, 3),
                "image_size": image_size,
            }
            out_dims.append(modalities_conf[col]["out_dim"])
        elif col_type in {"text", "text_path"}:
            text_out_dim = out_dim_map.get("text", 384)
            modalities_conf[col] = {
                "type": "text",
                "backbone": backbone_map.get("text", "all-MiniLM-L6-v2"),
                "out_dim": text_out_dim,
            }
            out_dims.append(text_out_dim)
        elif col_type in {"numeric", "categorical"}:
            tabular_columns.append(col)
        # add elif for future modalities (audio, video, etc.)

    # Combine tabular columns into a single "tabular" modality
    if tabular_columns:
        tabular_out_dim = out_dim_map.get("tabular", len(tabular_columns))
        modalities_conf["tabular"] = {
            "type": "tabular",
            "backbone": backbone_map.get("tabular", None),
            "out_dim": tabular_out_dim,
            "input_dim": len(tabular_columns),
            "columns": tabular_columns,
        }
        out_dims.append(tabular_out_dim)

    # Fusion output dim
    if fusion_out_dim is None:
        if fusion_type == "concat":
            fusion_out_dim = sum(out_dims)
        else:
            fusion_out_dim = max(out_dims) if out_dims else 0

    config = {
        "modalities": modalities_conf,
        "fusion": {
            "type": fusion_type,
            "out_dim": fusion_out_dim,
            "use_norm": use_norm,
        },
        "head": {
            "type": head_type,
            "num_classes": num_classes,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "activation": activation,
            "use_batchnorm": use_batchnorm,
        },
        "target_column": target_column,
    }
    return config


if __name__ == "__main__":
    # --- USAGE EXAMPLE ---
    selected_columns = ["Age", "Sex", "modality 1", "modality 2"]
    column_types = {'ID': 'numeric', 'N_Days': 'numeric', 'Status': 'categorical', 'Drug': 'categorical', 'Age': 'numeric', 'Sex': 'categorical', 'Ascites': 'categorical', 'Hepatomegaly': 'categorical', 'Spiders': 'categorical', 'Edema': 'categorical', 'Bilirubin': 'numeric', 'Cholesterol': 'numeric', 'Albumin': 'numeric', 'Copper': 'numeric', 'Alk_Phos': 'numeric', 'SGOT': 'numeric', 'Tryglicerides': 'numeric', 'Platelets': 'numeric', 'Prothrombin': 'numeric', 'Stage': 'numeric', 'm1': 'image_path', 'my text': 'text_path', 'm2': 'image_path', 'modality 1': 'image_path', 'modality 2': 'image_path'}
    target_column = "Edema"
    
    config = build_universal_multimodal_config(
        selected_columns=selected_columns,
        column_types=column_types,
        target_column=target_column,
        num_classes=3 # len(dataset.id2label)
    )

    print(config)