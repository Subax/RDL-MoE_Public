import os
import torch

DATA_DIR = "Submission/data"
CLINICAL_PATH = os.path.join(DATA_DIR, "sample_clinical.csv")
RADIOMICS_PATH = os.path.join(DATA_DIR, "sample_radiomics.csv")
OUTPUT_DIR = "Submission/results"

FIXED_N_FEATURES = 25
RANDOM_SEED = 42

MAX_EPOCHS = 1000
PATIENCE = 70
LEARNING_RATE = 0.005
RATIO_LAMBDA = 0.05

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
