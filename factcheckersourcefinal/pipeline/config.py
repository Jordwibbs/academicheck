import os
from dotenv import load_dotenv

load_dotenv()

# Model
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_PATH = "models/adapters/factchecker-lora"

# Embedding
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Paths
CHROMA_DIR = "db/chroma"
SQLITE_PATH = "db/papers.db"
RAW_DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# RAG
TOP_K_CHUNKS = 5
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Training
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TRAIN_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 2048

# Inference
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.1