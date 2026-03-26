#!/usr/bin/env python
"""
Main script to run EGD-based attacks on LLMs
"""

import argparse
import json
import os
import torch
from tqdm import tqdm

import config
from helper import (
    get_model_path,
    load_model_and_tokenizer,
    get_embedding_matrix,
    get_nonascii_toks,
    parse_csv
)
from attack import run_single_behavior_attack, generate_output
from behavior import Behavior

def main():
    parser = argparse.ArgumentParser(description="Run EGD-based LLM jailbreak attack")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["Llama2", "Falcon", "MPT", "Vicuna", "Mistral"],
                        help="Name of the target model")
    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=["AdvBench", "HarmBench", "JailbreakBench", "MaliciousInstruct"],
                        help="Name of the dataset")
    parser.add_argument("--num_behaviors", type=int, default=10,
                        help="Number of behaviors to attack")
    parser.add_argument("--num_steps", type=int, default=config.DEFAULT_NUM_STEPS,
                        help="Number of optimization steps")
    parser.add_argument("--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE,
                        help="Batch size (not used in single-behavior mode)")
    parser.add_argument("--step_size", type=float, default=config.DEFAULT_STEP_SIZE,
                        help="Learning rate")
    parser.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR,
                        help="Output directory")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    device = config.DEVICE
    if config.SEED is not None:
        torch.manual_seed(config.SEED)

    print(f"Loading model: {args.model_name}")
    model_path = get_model_path(args.model_name)
    model, tokenizer = load_model_and_tokenizer(
        model_path,
        low_cpu_mem_usage=True,
        use_cache=False,
        device=device
    )

    embed_weights = get_embedding_matrix(model)
    non_ascii_toks_tensor = get_nonascii_toks(tokenizer, device)

    dataset_path = config.DATASET_PATHS[args.dataset_name]
    harmful_behaviors = parse_csv(dataset_path)[:args.num_behaviors]

    output_json = os.path.join(args.output_dir, f"results_{args.model_name}_{args.dataset_name}_{args.num_behaviors}behaviors.json")
    output_jsonl = os.path.join(args.output_dir, f"results_{args.model_name}_{args.dataset_name}_{args.num_behaviors}behaviors.jsonl")

    # Pass the actual template string based on target model
    model_type_map = {"Llama2": "llama-2", "Mistral": "mistral", "Falcon": "falcon", "Vicuna": "vicuna", "MPT": "mpt"}
    model_type_str = model_type_map.get(args.model_name, "llama-2")

    results = []

    for idx, row in enumerate(tqdm(harmful_behaviors, desc="Attacking behaviors")):
        if len(row) != 2:
            continue

        user_prompt, target = row
        print(f"\n{'=' * 80}")
        print(f"Behavior {idx + 1}/{len(harmful_behaviors)}: {user_prompt}")
        print(f"Target: {target}")

        try:
            # Note the 4 variables returned now includes 'manager'
            effective_adv_one_hot, best_loss, best_epoch, manager = run_single_behavior_attack(
                model=model,
                tokenizer=tokenizer,
                user_prompt=user_prompt,
                target=target,
                embed_weights=embed_weights,
                non_ascii_toks_tensor=non_ascii_toks_tensor,
                device=device,
                num_steps=args.num_steps,
                step_size=args.step_size,
                model_type=model_type_str
            )

            # Route the manager directly to ensure generation uses proper format
            output, adv_token_ids = generate_output(
                model=model,
                tokenizer=tokenizer,
                manager=manager,
                adv_one_hot=effective_adv_one_hot,
                device=device,
                max_length=config.DEFAULT_NUM_TOKENS
            )

            adv_suffix = tokenizer.decode(adv_token_ids.cpu().numpy())

            print(f"\nAdversarial Suffix: {adv_suffix}")
            print(f"Best Loss: {best_loss:.4f} at epoch {best_epoch}")
            print(f"Generated Output: {output[:200]}...")

            result = {
                "harmful_behavior": user_prompt,
                "target": target,
                "suffix": adv_suffix,
                "suffix_token_ids": adv_token_ids.cpu().numpy().tolist(),
                "best_loss": float(best_loss),
                "best_epoch": int(best_epoch),
                "output": output
            }
            results.append(result)

            behavior = Behavior(user_prompt, adv_suffix, output, "", "")
            with open(output_jsonl, 'a') as f:
                f.write(json.dumps(behavior.to_dict()) + '\n')

        except Exception as e:
            print(f"Error processing behavior: {e}")
            continue

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}\nAttack complete!")

if __name__ == "__main__":
    main()