"""
Configuration file for EGD-based LLM attack
"""

# Device configuration
DEVICE = "cuda:0"
SEED = 42

# Model paths (update these to your local paths)
MODEL_PATHS = {
    "Llama2": "../LLMJailbreak/models/Llama-2-7b-chat-hf",
    "Falcon": "/path/to/falcon-7b-instruct",
    "MPT": "/path/to/mpt-7b-chat",
    "Vicuna": "/path/to/vicuna-7b-v1.5",
    "Mistral": "/path/to/Mistral-7B-Instruct-v0.3"
}

# Dataset paths
DATASET_PATHS = {
    "AdvBench": "./data/AdvBench/harmful_behaviors.csv",
    "HarmBench": "./data/HarmBench/harmful_behaviors.csv",
    "JailbreakBench": "./data/JailbreakBench/harmful_behaviors.csv",
    "MaliciousInstruct": "./data/MaliciousInstruct/harmful_behaviors.csv"
}

# Attack parameters
DEFAULT_NUM_STEPS = 200
DEFAULT_STEP_SIZE = 0.1
DEFAULT_BATCH_SIZE = 10
DEFAULT_NUM_TOKENS = 300

# Regularization parameters
INITIAL_ENTROPY_COEFF = 1e-5
FINAL_ENTROPY_COEFF = 1e-3
INITIAL_PEAKINESS_COEFF = 1e-5
FINAL_PEAKINESS_COEFF = 1e-3

# Optimizer parameters
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-4
GRAD_CLIP_MAX_NORM = 1.0

# Output directories
OUTPUT_DIR = "./outputs"
LOG_DIR = "./logs"
