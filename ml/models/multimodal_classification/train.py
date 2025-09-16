import time
import logging
from typing import Optional
import pandas as pd
from models.multimodal_classification.configs.config_creator import build_universal_multimodal_config
from models.multimodal_classification.dataset_builder import MultiModalDataset
from models.multimodal_classification.encoders.tabular_utils import TabularNormalizer
from models.multimodal_classification.models.multimodal import UniversalMultimodalClassification
from models.schemas import DLTrainingParameters, TrainingConfiguration, TrainingReport, Artifacts, ClassificationMetrics
from models.multimodal_classification.utils import get_acc_and_confmatrix, clean_text_columns, precision_recall_f1_from_cm
logger = logging.getLogger(__name__)


def train(params: DLTrainingParameters) -> tuple[TrainingReport, Optional[Artifacts]]:
    import torch
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from tqdm import tqdm
    from torch.utils.data import DataLoader, random_split
    
    # 1. Unpack parameters and paths
    training_params = params.configuration
    assets_paths = {asset['key']: asset['local_path'] for asset in params.assets_paths}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load and filter the full dataset for complete rows
    full_df = pd.read_csv(assets_paths["train_file"])
    full_df = clean_text_columns(
        df=full_df,
        column_types=params.column_types,
        selected_columns=params.feature_columns
    )
    base_dataset = MultiModalDataset(
        df=full_df,
        train_folder=assets_paths["train_folder"],
        selected_columns=params.feature_columns,
        column_types=params.column_types,
        target_column=params.target_column,
    )
    train_subset, val_subset = random_split(base_dataset, [1 - params.validation_split, params.validation_split])
    
    # 3. Get DataFrame of only the training samples
    train_df = base_dataset.df.iloc[train_subset.indices].reset_index(drop=True)
    val_df = base_dataset.df.iloc[val_subset.indices].reset_index(drop=True)

    # 4. Fit tabular and categorical normalization/encoding only on train
    cat_cols = [col for col in params.feature_columns if params.column_types[col] == "categorical"]
    # Build cat_maps for each categorical column
    cat_maps = {}
    for col in cat_cols:
        uniq = sorted(train_df[col].dropna().unique())
        cat_maps[col] = {cat: i for i, cat in enumerate(uniq)}
    # Tabular normalization for numeric+categorical features
    tabnorm = TabularNormalizer()
    tabnorm.fit(
        df=train_df,
        column_types=params.column_types,
        cat_maps=cat_maps,
        selected_columns=params.feature_columns,
    )

    # 5. Create train/val Datasets using fitted normalizer and categorical maps
    train_dataset = MultiModalDataset(
        df=train_df,
        train_folder=assets_paths["train_folder"],
        column_types=params.column_types,
        selected_columns=params.feature_columns,
        target_column=params.target_column,
        tabular_normalizer=tabnorm
    )
    val_dataset = MultiModalDataset(
        df=val_df,
        train_folder=assets_paths["train_folder"],
        selected_columns=params.feature_columns,
        column_types=params.column_types,
        target_column=params.target_column,
        tabular_normalizer=tabnorm
    )

    # 6. DataLoaders
    train_loader = DataLoader(train_dataset, training_params.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, training_params.batch_size, shuffle=False, num_workers=0, drop_last=True)
    num_classes = len(base_dataset.label2id)

    # 7. Model Building
    model_config = build_universal_multimodal_config(
        params.feature_columns,
        params.column_types,
        params.target_column,
        num_classes=num_classes
    )
    model = UniversalMultimodalClassification(model_config).to(device)

    # 8. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), training_params.learning_rate, weight_decay=0.0)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val_acc = float(0.0)
    best_model_state = None
    epoch_logs = []

    for epoch in range(training_params.epochs):
        model.train()
        train_losses = []
        train_outputs, train_targets = [], []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_params.epochs}", leave=False)
        for batch in progress_bar:
            inputs, targets = batch
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.compute_loss(outputs, targets)
            loss.backward()

            # Compute grad norm (optional)
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item()
            grad_norm = grad_norm ** 0.5

            optimizer.step()

            train_losses.append(loss.item())
            train_outputs.append(outputs.detach().cpu())
            train_targets.append(targets.detach().cpu())

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_outputs_all = torch.cat(train_outputs, dim=0)
        train_targets_all = torch.cat(train_targets, dim=0)
        train_acc, _ = get_acc_and_confmatrix(train_outputs_all, train_targets_all, num_classes=num_classes)

        if (epoch + 1) % training_params.eval_steps == 0:
            model.eval()
            val_losses = []
            val_outputs, val_targets = [], []
            t0 = time.time()
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch
                    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
                    targets = targets.to(device)

                    outputs = model(inputs)
                    loss = model.compute_loss(outputs, targets)
                    val_losses.append(loss.item())
                    val_outputs.append(outputs.cpu())
                    val_targets.append(targets.cpu())
            eval_runtime = time.time() - t0
            eval_steps_per_second = len(val_loader) / eval_runtime if eval_runtime > 0 else 0
            eval_samples_per_second = len(val_targets) / eval_runtime if eval_runtime > 0 else 0

            avg_val_loss = sum(val_losses) / len(val_losses)
            val_outputs_all = torch.cat(val_outputs, dim=0)
            val_targets_all = torch.cat(val_targets, dim=0)
            val_acc, val_cm = get_acc_and_confmatrix(val_outputs_all, val_targets_all, num_classes=num_classes)
            scheduler.step(avg_val_loss)

            # Record current learning rate (if using one group)
            lr = optimizer.param_groups[0]["lr"]

            # Record all metrics
            log_row = {
                "loss": sum(train_losses) / len(train_losses),
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
                "grad_norm": grad_norm,
                "val_runtime": eval_runtime,
                "val_accuracy": float(val_acc),
                "learning_rate": lr,
                "accuracy": float(train_acc),
                "val_confusion_matrix": val_cm,
                "val_steps_per_second": round(eval_steps_per_second, 3),
                "val_samples_per_second": round(eval_samples_per_second, 3),
            }
            epoch_logs.append(log_row)

            logger.info(log_row)

            # Save best model
            improved = val_acc > best_val_acc
            if improved:
                best_val_acc = val_acc
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

                precision, recall, f1_score = precision_recall_f1_from_cm(val_cm)

                metrics=ClassificationMetrics(
                    accuracy = float(train_acc),
                    val_accuracy = float(val_acc),
                    loss = sum(train_losses) / len(train_losses),
                    val_loss = avg_val_loss,
                    precision = precision,
                    recall = recall,
                    f1_score = f1_score,
                    confusion_matrix = val_cm
                    )


    # Load best model weights before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Loaded best model weights based on validation metric.")
    
    training_report = TrainingReport(
        logs=epoch_logs,
        metrics=metrics
    )

    artifacts = Artifacts(model.state_dict(), model_type='pt', id2label=base_dataset.id2label)

    return training_report, artifacts


if __name__ == "__main__":
    # Example usage
    params = DLTrainingParameters(
        model_name="mutlimodal_classification",
        feature_columns=["Age", "Tryglicerides", "modality 1", "modality 2"],
        column_types={'ID': 'other', 'N_Days': 'numeric', 'Status': 'categorical', 'Drug': 'categorical', 'Age': 'numeric', 'Sex': 'categorical', 'Ascites': 'categorical', 'Hepatomegaly': 'categorical', 'Spiders': 'categorical', 'Edema': 'categorical', 'Bilirubin': 'numeric', 'Cholesterol': 'numeric', 'Albumin': 'numeric', 'Copper': 'numeric', 'Alk_Phos': 'numeric', 'SGOT': 'numeric', 'Tryglicerides': 'numeric', 'Platelets': 'numeric', 'Prothrombin': 'numeric', 'Stage': 'numeric', 'm1': 'image_path', 'my text': 'text_path', 'm2': 'image_path', 'modality 1': 'image_path', 'modality 2': 'image_path'},
        target_column="Edema",
        validation_split=0.25,
        assets_paths=[{"key": "train_folder", "local_path": "/home/meysam/multimodal_demo_files/multimodal/dummy_multiimage_testset"},
                      {"key": "train_file", "local_path": "/home/meysam/multimodal_demo_files/multimodal/cirrhosis_example_file_multimodal.csv"}],
        training_job_id=160,
        configuration=TrainingConfiguration(
            learning_rate=0.0002,
            epochs=2,
            batch_size=8,
            early_stopping=True,
            early_stopping_patience=5,
            random_seed=42,
            eval_steps=1
        )
        )
    
    training_report, artifacts = train(params)

    print(f"Training Report: {training_report}")