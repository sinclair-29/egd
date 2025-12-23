"""
Helper functions for model loading and data processing
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, FalconForCausalLM, MistralForCausalLM
import config


def get_model_path(model_name):
    """Get the local path for a model"""
    if model_name not in config.MODEL_PATHS:
        raise ValueError(f"Unknown model name: {model_name}")
    return config.MODEL_PATHS[model_name]


def load_model_and_tokenizer(model_path, device=config.DEVICE, **kwargs):
    """Load model and tokenizer from path"""
    # Determine dtype based on model
    if "llama" in model_path.lower() or "vicuna" in model_path.lower():
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=dtype, 
        trust_remote_code=False, 
        **kwargs
    ).to(device).eval()
    
    # Special handling for MPT tokenizer
    if "mpt" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=False, 
            use_fast=False
        )
    
    # Set padding token
    if "llama-2" in model_path.lower():
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "left"
    elif "falcon" in model_path.lower():
        tokenizer.padding_side = "left"
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def get_embedding_matrix(model):
    """Extract embedding matrix from model"""
    if isinstance(model, (LlamaForCausalLM, MistralForCausalLM)):
        return model.model.embed_tokens.weight
    elif isinstance(model, FalconForCausalLM):
        return model.get_input_embeddings().weight.data
    else:
        # Generic fallback
        return model.get_input_embeddings().weight.data


def get_tokens(input_string, tokenizer, device):
    """Tokenize input string"""
    return torch.tensor(tokenizer(input_string)["input_ids"], device=device)


def create_one_hot_and_embeddings(tokens, embed_weights, device):
    """Create one-hot encodings and embeddings from tokens"""
    one_hot = torch.zeros(
        tokens.shape[0], 
        embed_weights.shape[0], 
        device=device, 
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        tokens.unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=device, dtype=embed_weights.dtype),
    )
    embeddings = (one_hot @ embed_weights).unsqueeze(0).data
    return one_hot, embeddings


def get_nonascii_toks(tokenizer, device):
    """Get list of non-ASCII token IDs"""
    def is_ascii(s):
        return s.isascii() and s.isprintable()
    
    non_ascii_toks = []
    for i in range(0, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            non_ascii_toks.append(i)
    
    # Add special tokens
    for token_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, 
                     tokenizer.pad_token_id, tokenizer.unk_token_id]:
        if token_id is not None:
            non_ascii_toks.append(token_id)
    
    return torch.tensor(non_ascii_toks).to(device)


def get_masked_one_hot_adv(one_hot_adv, non_ascii_toks_tensor):
    """Mask out non-ASCII tokens from adversarial one-hot"""
    top_token_ids_tensor_2d = non_ascii_toks_tensor.unsqueeze(0).repeat(20, 1)
    mask = torch.ones_like(one_hot_adv, dtype=torch.float16)
    mask.scatter_(1, top_token_ids_tensor_2d, 0.0)
    masked_one_hot_adv = one_hot_adv * mask
    return masked_one_hot_adv


def parse_csv(csv_path):
    """Parse CSV file containing harmful behaviors"""
    import csv
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        rows = list(reader)
    return rows
