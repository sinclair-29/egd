"""
Main attack implementation using EGD with Adam optimizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm
import config
from helper import create_one_hot_and_embeddings, get_masked_one_hot_adv
from prompt_manager import PromptManager  # assume the provided PromptManager is in a separate file
from fastchat.model import get_conversation_template

def calc_loss(model, embeddings_user, embeddings_adv, embeddings_target, targets):
    """Calculate cross-entropy loss"""
    full_embeddings = torch.hstack([embeddings_user, embeddings_adv, embeddings_target])
    logits = model(inputs_embeds=full_embeddings).logits
    loss_slice_start = len(embeddings_user[0]) + len(embeddings_adv[0])
    loss = nn.CrossEntropyLoss()(logits[0, loss_slice_start - 1 : -1, :], targets)
    return loss, logits[:, loss_slice_start-1:, :]

def generate_output_with_embeddings(model, tokenizer, user_embeddings, adv_embeddings, device, max_length=300):
    """
    Generate model output using embeddings directly.
    user_embeddings: tensor of shape (1, user_len, hidden_dim)
    adv_embeddings: tensor of shape (1, adv_len, hidden_dim)
    """
    combined_embeds = torch.cat([user_embeddings, adv_embeddings], dim=1)  # (1, total_len, hidden_dim)
    attention_mask = torch.ones(combined_embeds.shape[:2], dtype=torch.long, device=device)

    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", message=".*`do_sample` is set to `False`.*")

    generated_output = model.generate(
        inputs_embeds=combined_embeds,
        attention_mask=attention_mask,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False
    )

    generated_output_string = tokenizer.decode(
        generated_output[0][:].cpu().numpy(),
        skip_special_tokens=True
    ).strip()
    return generated_output_string


class EGDwithAdamOptimizer(torch.optim.Optimizer):
    """Custom optimizer combining EGD with Adam"""
    
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-4):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        super(EGDwithAdamOptimizer, self).__init__(params, defaults)
        
        # Initialize state variables
        for group in self.param_groups:
            for param in group['params']:
                self.state[param] = {
                    'm': torch.zeros_like(param),
                    'v': torch.zeros_like(param),
                    't': 0
                }
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad
                state = self.state[param]
                
                # Retrieve state variables
                m = state['m']
                v = state['v']
                t = state['t']
                
                # Increment time step
                t += 1
                state['t'] = t
                
                # Update biased first and second moment estimates
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * torch.pow(grad, 2)
                state['m'] = m
                state['v'] = v
                
                # Bias correction
                m_hat = m / (1 - math.pow(beta1, t))
                v_hat = v / (1 - math.pow(beta2, t))
                
                # Adam-like modified gradient
                modified_grad = m_hat / (torch.sqrt(v_hat) + eps)
                
                with torch.no_grad():
                    # Exponentiated Gradient Descent update
                    param.mul_(torch.exp(-lr * modified_grad))
                    param.clamp_(min=1e-12, max=1e12)
                    
                    # Normalize to probability simplex
                    if param.dim() > 1:
                        row_sums = param.sum(dim=1, keepdim=True) + 1e-10
                        param.div_(row_sums)
                    else:
                        param.div_(param.sum() + 1e-10)
        
        return loss

def load_conversation_template(template_name: str):
    conv_template = get_conversation_template(template_name)
    if conv_template.name == 'llama-2':
        conv_template.system_message = "You are a helpful assistant."
    # add other models as needed
    return conv_template


def run_single_behavior_attack(
    model,
    tokenizer,
    user_prompt,      # instruction
    target,           # desired response
    embed_weights,
    non_ascii_toks_tensor,
    device,
    num_steps=200,
    step_size=0.1,
    output_freq=50,
    model_type="llama-2"   # new argument to choose chat template
):
    # Load conversation template
    conv_template = load_conversation_template(model_type)

    # Initial adversarial suffix placeholder (random tokens)
    adv_len = 20  # length of adversarial suffix
    initial_adv_str = "! " * adv_len   # dummy string, will be replaced by learnable embeddings

    # Create PromptManager to build the full prompt and slices
    manager = PromptManager(
        tokenizer=tokenizer,
        conv_template=conv_template,
        instruction=user_prompt,
        target=target,
        adv_string=initial_adv_str
    )

    # Get full token IDs with placeholder adv string
    full_input_ids = manager.get_input_ids(adv_string=initial_adv_str).to(device)

    # Determine slices from the manager
    adv_slice = manager.get_adv_slice()
    target_slice = manager.get_loss_slice()  # note: loss is computed on target tokens after assistant role
    # The loss slice in the reference code includes the target tokens (the desired completion)

    # Split the full input into three parts:
    # 1. Fixed tokens before the adversarial suffix
    # 2. Adversarial suffix (will be learnable)
    # 3. Fixed tokens after the adversarial suffix (including assistant role and target)
    prefix_ids = full_input_ids[:adv_slice.start]
    suffix_ids = full_input_ids[adv_slice.stop:]

    # Create fixed embeddings for prefix and suffix
    with torch.no_grad():
        prefix_embeds = model.get_input_embeddings()(prefix_ids.unsqueeze(0))
        suffix_embeds = model.get_input_embeddings()(suffix_ids.unsqueeze(0))

    # Initialize learnable one-hot for adversarial suffix
    one_hot_adv = F.softmax(
        torch.rand(adv_len, embed_weights.shape[0], dtype=torch.float16).to(device),
        dim=1
    ).to(embed_weights.dtype)
    one_hot_adv.requires_grad_()

    # Optimizer and scheduler
    optimizer = EGDwithAdamOptimizer([one_hot_adv], lr=step_size)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=50
    )

    best_loss = np.inf
    best_epoch = 0
    best_one_hot = None

    # Training loop
    for epoch in tqdm(range(num_steps), desc="Optimizing"):
        optimizer.zero_grad()

        # Compute adversarial embeddings from current one_hot
        adv_embeds = (one_hot_adv @ embed_weights).unsqueeze(0)  # shape (1, adv_len, hidden)

        # Concatenate full embedding tensor
        full_embeds = torch.cat([prefix_embeds, adv_embeds, suffix_embeds], dim=1)  # (1, total_len, hidden)

        # Forward pass
        logits = model(inputs_embeds=full_embeds).logits  # (1, total_len, vocab)

        # Compute cross-entropy loss on the target tokens
        # The loss slice indices are relative to the original full_input_ids, which correspond to positions in full_embeds
        loss_slice = target_slice
        # Adjust for the fact that full_embeds length may differ if prefix/adv/suffix lengths changed? Actually they match exactly because we used the same slices.
        # But careful: the loss slice might include the entire assistant response (including target). In the reference, loss_slice is slice(assistant_role_slice.stop-1, ...)
        # That slice covers the target tokens.
        target_logits = logits[0, loss_slice.start-1:loss_slice.stop-1, :]  # shift by -1 because logits at position i predict token i+1
        target_ids = full_input_ids[loss_slice.start:loss_slice.stop]
        cross_entropy_loss = F.cross_entropy(target_logits, target_ids)

        continuous_loss = cross_entropy_loss.detach().cpu().item()

        if math.isnan(continuous_loss):
            print("NaN detected, stopping")
            break

        # Regularization (entropy and peakiness) as before
        progress = epoch / (num_steps - 1 + 1e-12)
        entropy_coeff = config.INITIAL_ENTROPY_COEFF * (
            config.FINAL_ENTROPY_COEFF / config.INITIAL_ENTROPY_COEFF
        ) ** progress
        peakiness_coeff = config.INITIAL_PEAKINESS_COEFF * (
            config.FINAL_PEAKINESS_COEFF / config.INITIAL_PEAKINESS_COEFF
        ) ** progress

        one_hot_adv_float32 = one_hot_adv.to(dtype=torch.float32)
        log_one_hot = torch.log(one_hot_adv_float32 + 1e-12)
        entropy_term = (-one_hot_adv_float32 * (log_one_hot - 1)).sum() * entropy_coeff
        peakiness_term = -torch.log(torch.max(one_hot_adv, dim=1).values + 1e-12).sum() * peakiness_coeff

        regularized_loss = cross_entropy_loss - entropy_term + peakiness_term

        # Backward
        regularized_loss.backward()
        torch.nn.utils.clip_grad_norm_([one_hot_adv], max_norm=config.GRAD_CLIP_MAX_NORM)
        optimizer.step()
        scheduler.step(continuous_loss)

        # Optionally generate output from soft embeddings (using helper function from previous answer)
        if output_freq and epoch % output_freq == 0:
            with torch.no_grad():
                # You can use the `generate_output_with_embeddings` function from earlier
                full_embeds_for_gen = torch.cat([prefix_embeds, adv_embeds, suffix_embeds], dim=1)
                # Generate response
                generated = model.generate(
                    inputs_embeds=full_embeds_for_gen,
                    max_length=full_embeds_for_gen.shape[1] + 50,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
                output_str = tokenizer.decode(generated[0], skip_special_tokens=True)
                print(f"\n--- Epoch {epoch} | Generated: ---\n{output_str}\n---\n")

        # Discretization and tracking best loss
        with torch.no_grad():
            masked_one_hot = get_masked_one_hot_adv(one_hot_adv, non_ascii_toks_tensor)
            adv_token_ids = masked_one_hot.argmax(dim=1)
            # Build full embedding with discrete suffix
            discrete_adv_embeds = (F.one_hot(adv_token_ids, embed_weights.shape[0]).to(embed_weights.dtype) @ embed_weights).unsqueeze(0)
            full_embeds_discrete = torch.cat([prefix_embeds, discrete_adv_embeds, suffix_embeds], dim=1)
            logits_discrete = model(inputs_embeds=full_embeds_discrete).logits
            target_logits_disc = logits_discrete[0, loss_slice.start-1:loss_slice.stop-1, :]
            disc_loss = F.cross_entropy(target_logits_disc, target_ids).detach().cpu().item()

            if disc_loss < best_loss:
                best_loss = disc_loss
                best_epoch = epoch
                best_one_hot = masked_one_hot.clone()

    return best_one_hot, best_loss, best_epoch

def generate_output(model, tokenizer, user_prompt_tokens, adv_one_hot, device, max_length=300):
    """Generate model output with adversarial suffix"""
    adv_token_ids = adv_one_hot.argmax(dim=1)
    final_string_ids = torch.hstack([user_prompt_tokens, adv_token_ids])
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", message=".*`do_sample` is set to `False`.*")
    
    generated_output = model.generate(
        final_string_ids.unsqueeze(0),
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False
    )
    
    generated_output_string = tokenizer.decode(
        generated_output[0][:].cpu().numpy(), 
        skip_special_tokens=True
    ).strip()
    
    return generated_output_string, adv_token_ids
