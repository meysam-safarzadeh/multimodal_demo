from models.multimodal.train import train_model


def main():
    print("Starting training...")
    accuracy, loss, classes = train_model()
    print(f"Results: accuracy={accuracy:.4f}, loss={loss:.4f}, classes={classes}")


if __name__ == "__main__":
    main()