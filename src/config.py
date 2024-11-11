from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    CONCEPTS_FILE = BASE_DIR / "filtered_concepts.json"
    
    # Training settings
    BATCH_SIZE = 64
    NUM_EPOCHS = 3
    LEARNING_RATE = 0.01
    L1_LAMBDA = 0.001  # Sparsity regularization

config = Config()
