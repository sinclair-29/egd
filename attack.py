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


def calc_loss(model, embeddings_user, embeddings_adv, embeddings_target, targets):
    """Calculate cross-entropy loss"""
    full_embeddings = torch.hstack([embeddings_user, embeddings_adv, embeddings_target])
    logits = model(inputs_embeds=full_embeddings).logits
    loss_slice_start = len(embeddings_user[0]) + len(embeddings_adv[0])
    loss = nn.CrossEntropyLoss()(logits[0, loss_slice_start - 1 : -1, :], targets)
    return loss, logits[:, loss_slice_start-1:, :]


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


def run_single_behavior_attack(
    model, 
    tokenizer, 
    user_prompt, 
    target, 
    embed_weights,
    non_ascii_toks_tensor,
    device,
    num_steps=200,
    step_size=0.1
):
    """Run attack on a single behavior"""

    
    # Tokenize inputs
    user_prompt_tokens = torch.tensor(tokenizer(user_prompt)["input_ids"], device=device)
    target_tokens = torch.tensor(tokenizer(target)["input_ids"], device=device)[1:]
    
    # Create embeddings
    _, embeddings_user = create_one_hot_and_embeddings(user_prompt_tokens, embed_weights, device)
    one_hot_target, embeddings_target = create_one_hot_and_embeddings(target_tokens, embed_weights, device)
    
    # Initialize adversarial one-hot
    one_hot_adv = F.softmax(
        torch.rand(20, embed_weights.shape[0], dtype=torch.float16).to(device), 
        dim=1
    ).to(embed_weights.dtype)
    one_hot_adv.requires_grad_()
    
    # Initialize optimizer and scheduler
    optimizer = EGDwithAdamOptimizer([one_hot_adv], lr=step_size)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=50
    )
    
    # Track best loss
    best_disc_loss = np.inf
    best_loss_at_epoch = 0
    effective_adv_one_hot = None
    
    # Training loop
    for epoch_no in tqdm(range(num_steps), desc="Optimizing"):
        optimizer.zero_grad()
        
        # Forward pass
        embeddings_adv = (one_hot_adv @ embed_weights).unsqueeze(0)
        cross_entropy_loss, _ = calc_loss(
            model, embeddings_user, embeddings_adv, embeddings_target, one_hot_target
        )
        continuous_loss = cross_entropy_loss.detach().cpu().item()
        
        # Check for NaN
        if math.isnan(continuous_loss):
            print("NaN detected, stopping")
            break
        
        # Compute regularization terms
        eps = 1e-12
        progress = epoch_no / (num_steps - 1 + eps)
        
        # Entropy regularization (decay)
        entropy_coeff = config.INITIAL_ENTROPY_COEFF * (
            config.FINAL_ENTROPY_COEFF / config.INITIAL_ENTROPY_COEFF
        ) ** progress
        
        one_hot_adv_float32 = one_hot_adv.to(dtype=torch.float32)
        log_one_hot = torch.log(one_hot_adv_float32 + eps)
        entropy_term = (-one_hot_adv_float32 * (log_one_hot - 1)).sum() * entropy_coeff
        
        # Peakiness regularization (growth)
        peakiness_coeff = config.INITIAL_PEAKINESS_COEFF * (
            config.FINAL_PEAKINESS_COEFF / config.INITIAL_PEAKINESS_COEFF
        ) ** progress
        
        peakiness_term = -torch.log(torch.max(one_hot_adv, dim=1).values + eps).sum()
        peakiness_term *= peakiness_coeff
        
        # Total regularized loss
        regularized_loss = cross_entropy_loss - entropy_term + peakiness_term
        
        # Backward pass
        regularized_loss.backward()
        torch.nn.utils.clip_grad_norm_([one_hot_adv], max_norm=config.GRAD_CLIP_MAX_NORM)
        
        # Update parameters
        optimizer.step()
        scheduler.step(continuous_loss)
        
        # Discretization
        with torch.no_grad():
            masked_one_hot_adv = get_masked_one_hot_adv(one_hot_adv, non_ascii_toks_tensor)
            adv_token_ids = masked_one_hot_adv.argmax(dim=1)
            one_hot_discrete = F.one_hot(
                adv_token_ids, 
                num_classes=embed_weights.shape[0]
            ).to(embed_weights.dtype)
            
            embeddings_adv_discrete = (one_hot_discrete @ embed_weights).unsqueeze(0)
            disc_loss, _ = calc_loss(
                model, embeddings_user, embeddings_adv_discrete, 
                embeddings_target, one_hot_target
            )
            discrete_loss = disc_loss.detach().cpu().item()
            
            if discrete_loss < best_disc_loss:
                best_disc_loss = discrete_loss
                best_loss_at_epoch = epoch_no
                effective_adv_one_hot = one_hot_discrete.clone()
    
    return effective_adv_one_hot, best_disc_loss, best_loss_at_epoch


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
