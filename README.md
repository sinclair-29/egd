# EGD-Based LLM Jailbreak Attack (Simplified)

This is a simplified implementation of the Exponentiated Gradient Descent (EGD) based adversarial attack method for Large Language Models, as proposed in the research paper.

## Overview

This repository contains the core implementation for generating adversarial suffixes to jailbreak aligned LLMs using EGD with Adam optimizer.

## Key Features

- **EGD with Adam Optimizer**: Custom optimizer combining Exponentiated Gradient Descent with Adam's adaptive learning rates
- **Multi-prompt Attack**: Batch optimization across multiple harmful behaviors
- **Entropy Regularization**: Encourages exploration while maintaining discreteness
- **Support for Multiple Models**: Llama2, Falcon, MPT, Vicuna, Mistral

## Project Structure

```
egd-llm-attack-simplified/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── config.py                   # Configuration settings
├── helper.py                   # Utility functions
├── behavior.py                 # Behavior data class
├── attack.py                   # Main attack implementation
├── run_attack.py              # Script to run attacks
└── data/                      # Dataset directory
    └── AdvBench/
        └── harmful_behaviors.csv
```

## Installation

```bash
# Create a conda environment
conda create -n llm-attack python=3.8
conda activate llm-attack

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Single Behavior Attack

```bash
python run_attack.py \
    --model_name Llama2 \
    --dataset_name AdvBench \
    --num_behaviors 1 \
    --num_steps 200 \
    --batch_size 10
```

### Multiple Behaviors Attack

```bash
python run_attack.py \
    --model_name Llama2 \
    --dataset_name AdvBench \
    --num_behaviors 50 \
    --num_steps 200 \
    --batch_size 10
```

## Parameters

- `--model_name`: Target model (Llama2, Falcon, MPT, Vicuna, Mistral)
- `--dataset_name`: Dataset to use (AdvBench, HarmBench, JailbreakBench, MaliciousInstruct)
- `--num_behaviors`: Number of harmful behaviors to attack
- `--num_steps`: Number of optimization steps (default: 200)
- `--batch_size`: Batch size for multi-prompt attack (default: 10)
- `--step_size`: Learning rate (default: 0.1)

## Output

Results are saved in:
- `outputs/`: Generated adversarial suffixes and model responses
- `logs/`: Training statistics and loss curves

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{llm-egd-attack,
  title={Adversarial Attacks on LLMs using Exponentiated Gradient Descent},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project is for research purposes only. Please use responsibly.

## Acknowledgments

Based on the LLM-attacks framework and inspired by gradient-based adversarial attack methods.
