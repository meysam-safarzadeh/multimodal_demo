from models.multimodal_classification.train import train
from models.schemas import DLTrainingParameters, TrainingConfiguration


def main():
    # Example usage
    params = DLTrainingParameters(
        model_name="mutlimodal_classification",
        feature_columns=["Age", "Tryglicerides", "modality 1", "modality 2"],
        column_types={'ID': 'other', 'N_Days': 'numeric', 'Status': 'categorical', 'Drug': 'categorical', 'Age': 'numeric', 'Sex': 'categorical', 'Ascites': 'categorical', 'Hepatomegaly': 'categorical', 'Spiders': 'categorical', 'Edema': 'categorical', 'Bilirubin': 'numeric', 'Cholesterol': 'numeric', 'Albumin': 'numeric', 'Copper': 'numeric', 'Alk_Phos': 'numeric', 'SGOT': 'numeric', 'Tryglicerides': 'numeric', 'Platelets': 'numeric', 'Prothrombin': 'numeric', 'Stage': 'numeric', 'm1': 'image_path', 'my text': 'text_path', 'm2': 'image_path', 'modality 1': 'image_path', 'modality 2': 'image_path'},
        target_column="Edema",
        validation_split=0.25,
        assets_paths=[{"key": "train_folder", "local_path": "/home/meysam/multimodal_demo_files/multimodal/dummy_multiimage_testset"},
                      {"key": "train_file", "local_path": "/home/meysam/multimodal_demo_files/multimodal/cirrhosis_example_file_multimodal_filled_new.csv"}],
        training_job_id=160,
        configuration=TrainingConfiguration(
            learning_rate=0.0002,
            epochs=2,
            batch_size=2,
            early_stopping=True,
            early_stopping_patience=5,
            random_seed=42,
            eval_steps=1
        )
        )
    print("Starting training...")
    training_report, artifacts = train(params)
    print("Training completed.")
    # print(f"Results: accuracy={training_report.metrics['accuracy']:.4f}, loss={training_report.metrics['loss']:.4f}, classes={training_report.metrics['classes']}")


if __name__ == "__main__":
    main()