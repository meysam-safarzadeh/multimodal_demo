import time
import numpy as np


def train_model():
    """Dummy training function that generates random metrics."""
    time.sleep(7)  # simulate some training time
    np.random.seed(0)
    accuracy = np.random.rand()
    loss = np.random.rand()
    classes = np.random.randint(2, 10)  # e.g., between 2 and 10 classes

    print(f"Training complete. accuracy={accuracy:.4f}, loss={loss:.4f}, classes={classes}")
    return accuracy, loss, classes