# Quick Start Guide

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Update model paths in `config.py`:**
Edit the `MODEL_PATHS` dictionary to point to your local model directories.

```python
MODEL_PATHS = {
    "Llama2": "/your/path/to/Llama-2-7b-chat-hf",
    "Falcon": "/your/path/to/falcon-7b-instruct",
    # ... etc
}
```

## Running an Attack

### Basic Example
```bash
python run_attack.py \
    --model_name Llama2 \
    --dataset_name AdvBench \
    --num_behaviors 5 \
    --num_steps 200
```

### Full Options
```bash
python run_attack.py \
    --model_name Llama2 \
    --dataset_name AdvBench \
    --num_behaviors 10 \
    --num_steps 200 \
    --step_size 0.1 \
    --output_dir ./my_results
```

## Output Format

Results are saved in two formats:

1. **JSON** (`results_*.json`): Complete results with all metadata
2. **JSONL** (`results_*.jsonl`): Line-delimited format for easy parsing

Each result contains:
- `harmful_behavior`: The original harmful prompt
- `target`: The target completion
- `suffix`: The generated adversarial suffix
- `suffix_token_ids`: Token IDs of the suffix
- `best_loss`: Lowest loss achieved
- `best_epoch`: Epoch where best loss occurred
- `output`: Model's generated response

## Key Components

- **`config.py`**: All configuration parameters
- **`helper.py`**: Utility functions for model loading and data processing
- **`attack.py`**: Core attack implementation with EGD optimizer
- **`run_attack.py`**: Main script to run attacks
- **`behavior.py`**: Data class for storing results

## Method Overview

The attack uses **Exponentiated Gradient Descent (EGD) with Adam optimizer** to find adversarial suffixes that cause aligned LLMs to comply with harmful requests.

### Key Features:

1. **EGD Optimization**: Updates in probability simplex while maintaining valid distributions
2. **Adam Momentum**: Adaptive learning rates for faster convergence
3. **Entropy Regularization**: Encourages exploration in early stages
4. **Peakiness Regularization**: Promotes discreteness in later stages
5. **Non-ASCII Filtering**: Masks out non-printable tokens

### Algorithm Flow:

1. Initialize random adversarial suffix (20 tokens)
2. For each optimization step:
   - Compute cross-entropy loss
   - Add entropy and peakiness regularization
   - Update via EGD-Adam
   - Discretize to find best tokens
3. Generate final output with best suffix

## Troubleshooting

**CUDA out of memory:**
- Reduce `num_steps`
- Use smaller models
- Set `DEVICE = "cpu"` in config.py (slower)

**Model path errors:**
- Verify paths in `config.py`
- Ensure models are downloaded locally

**Import errors:**
- Check `requirements.txt` versions
- Ensure PyTorch is installed with CUDA support

## Research Use Only

This code is for research purposes only. Please use responsibly and follow ethical guidelines.
