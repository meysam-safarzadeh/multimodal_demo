import torch #TODO move to inside __init__ to avoid import errors in other models
import torch.nn as nn
from models.multimodal_classification.encoders.image_encoder import ImageEncoderClassification
from models.multimodal_classification.encoders.tabular_encoder import TabularEncoder
from models.multimodal_classification.encoders.text_encoder import TextEncoder
from models.multimodal_classification.fusion.concat import ConcatFeatureMaps, ConcatVectors
from models.multimodal_classification.heads.classifier_head import ClassifierHead


class UniversalMultimodalClassification(nn.Module):
    """
    Universal modular multimodal model, composable from config.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: dict specifying modalities, encoders, fusion, head, etc.
        Example config:
        {
            "modalities": {
                "rgb": {"type": "image", "backbone": "resnet50", "out_dim": 128, "in_channels": 3},
                "thermal": {"type": "image", "backbone": "resnet18", "out_dim": 128, "in_channels": 1},
                "tabular": {"type": "tabular", "out_dim": 64, "input_dim": 20, "columns": ['Age', 'Sex']}
                # ...add more as needed
            },
            "fusion": {"type": "concat", "out_dim": 256, "use_norm": True},
            "head": {"type": "classifier", "num_classes": 10, "hidden_dims": [512, 128], "dropout": 0.2, "activation": "gelu", "use_batchnorm": True}
        }
        """
        super().__init__()
        self.modalities = list(config["modalities"].keys())
        self.encoder_modules = nn.ModuleDict()
        self.loss_fn = nn.CrossEntropyLoss()  # Default loss, can be overridden in head
        encoder_output_dims = []

        # Build encoders for each modality
        self.head_conf = config["head"]
        for mod_name, mod_conf in config["modalities"].items():
            if mod_conf["type"] == "image" and self.head_conf["type"] == "classifier":
                encoder = ImageEncoderClassification(
                    backbone=mod_conf.get("backbone", "resnet18"),
                    pretrained=mod_conf.get("pretrained", True),
                    out_dim=mod_conf["out_dim"],
                    in_channels=mod_conf.get("in_channels", 3),
                    freeze_backbone=mod_conf.get("freeze_backbone", False),
                )
            elif mod_conf["type"] == "tabular":
                encoder = TabularEncoder(
                    in_dim=mod_conf["input_dim"],
                    out_dim=mod_conf["out_dim"]
                )
            elif mod_conf["type"] == "text":
                encoder = TextEncoder(
                    backbone=mod_conf.get("backbone", "all-MiniLM-L6-v2"),
                    out_dim=mod_conf["out_dim"]
                )
            # ...add more modality types as you develop them
            else:
                raise ValueError(f"Unknown modality type: {mod_conf['type']}")
            self.encoder_modules[mod_name] = encoder
            encoder_output_dims.append(mod_conf["out_dim"])

        # Build fusion module
        fusion_conf = config["fusion"]
        if fusion_conf["type"] == "concat" and self.head_conf["type"] == "classifier":
            self.fusion = ConcatVectors(
                input_dims=encoder_output_dims,
                out_dim=fusion_conf.get("out_dim", sum(encoder_output_dims)),
                use_norm=fusion_conf.get("use_norm", False)
            )
        elif fusion_conf["type"] == "concat" and self.head_conf["type"] == "detector":
            self.fusion = ConcatFeatureMaps()
        # ... add support for more fusion types

        # Build head module
        if self.head_conf["type"] == "classifier":
            self.head = ClassifierHead(
                in_dim=fusion_conf.get("out_dim", sum(encoder_output_dims)),
                num_classes=self.head_conf["num_classes"],
                hidden_dims=self.head_conf.get("hidden_dims", [512, 128]),
                activation=self.head_conf.get("activation", "relu"),
                dropout=self.head_conf.get("dropout", 0.0),
                use_batchnorm=self.head_conf.get("use_batchnorm", False),
            )
        # ... add support for other heads (regressor, sequence, etc.)

    def forward(self, inputs: dict) -> torch.Tensor:
        """
        Args:
            inputs: dict mapping modality name to input tensor
                e.g., {"rgb": tensor, "thermal": tensor, "tabular": tensor}
        Returns:
            Output tensor from head (e.g., logits for classifier)
        """
        features = []
        for mod in self.modalities:
            x = self.encoder_modules[mod](inputs[mod])
            features.append(x)
        fused = self.fusion(features)
        if self.head_conf["type"] == "detector":
            output = self.head(fused, inputs.get("rgb", None))
        else:
            output = self.head(fused)
        return output
    
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.head_conf["type"] == "classifier":
            return self.loss_fn(outputs, targets)
        elif self.head_conf["type"] == "detector":
            # For detection head, compute loss using the head's specific method
            return self.head.compute_loss(outputs, targets)
        else:
            raise ValueError(f"Unsupported head type for loss computation: {self.head_conf['type']}")

