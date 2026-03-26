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
from prompt_manager import PromptManager, load_conversation_template

class EGDwithAdamOptimizer(torch.optim.Optimizer):
    """Custom optimizer combining EGD with Adam"""

    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-4):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        super(EGDwithAdamOptimizer, self).__init__(params, defaults)

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
            lr, beta1, beta2, eps = group['lr'], group['beta1'], group['beta2'], group['eps']

            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]

                state['t'] += 1
                t = state['t']
                m, v = state['m'], state['v']

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * torch.pow(grad, 2)
                state['m'], state['v'] = m, v

                m_hat = m / (1 - math.pow(beta1, t))
                v_hat = v / (1 - math.pow(beta2, t))
                modified_grad = m_hat / (torch.sqrt(v_hat) + eps)

                with torch.no_grad():
                    param.mul_(torch.exp(-lr * modified_grad))
                    param.clamp_(min=1e-12, max=1e12)
                    if param.dim() > 1:
                        param.div_(param.sum(dim=1, keepdim=True) + 1e-10)
                    else:
                        param.div_(param.sum() + 1e-10)

        return loss

def run_single_behavior_attack(
    model, tokenizer, user_prompt, target, embed_weights,
    non_ascii_toks_tensor, device, num_steps=200, step_size=0.1,
    output_freq=10, model_type="llama-2"
):
    conv_template = load_conversation_template(model_type)

    adv_len_estimate = 20
    initial_adv_str = "! " * adv_len_estimate

    manager = PromptManager(
        tokenizer=tokenizer,
        conv_template=conv_template,
        instruction=user_prompt,
        target=target,
        adv_string=initial_adv_str
    )

    full_input_ids = manager.get_input_ids(adv_string=initial_adv_str).to(device)

    adv_slice = manager.get_adv_slice()
    loss_slice = manager.get_loss_slice()

    # FIX: Dynamically align adversarial tensor size with tokenizer output
    # so `full_embeds` precisely aligns with `loss_slice` indices.
    actual_adv_len = adv_slice.stop - adv_slice.start

    prefix_ids = full_input_ids[:adv_slice.start]
    suffix_ids = full_input_ids[adv_slice.stop:]

    with torch.no_grad():
        prefix_embeds = model.get_input_embeddings()(prefix_ids.unsqueeze(0))
        suffix_embeds = model.get_input_embeddings()(suffix_ids.unsqueeze(0))

    one_hot_adv = F.softmax(
        torch.rand(actual_adv_len, embed_weights.shape[0], dtype=torch.float16).to(device),
        dim=1
    ).to(embed_weights.dtype)
    one_hot_adv.requires_grad_()

    optimizer = EGDwithAdamOptimizer([one_hot_adv], lr=step_size)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=50
    )

    best_loss = np.inf
    best_epoch = 0
    best_one_hot = None

    for epoch in tqdm(range(num_steps), desc="Optimizing"):
        optimizer.zero_grad()

        adv_embeds = (one_hot_adv @ embed_weights).unsqueeze(0)
        full_embeds = torch.cat([prefix_embeds, adv_embeds, suffix_embeds], dim=1)

        logits = model(inputs_embeds=full_embeds).logits

        target_logits = logits[0, loss_slice.start-1:loss_slice.stop-1, :]
        target_ids = full_input_ids[loss_slice.start:loss_slice.stop]
        cross_entropy_loss = F.cross_entropy(target_logits, target_ids)

        continuous_loss = cross_entropy_loss.detach().cpu().item()

        if math.isnan(continuous_loss):
            print("NaN detected, stopping")
            break

        progress = epoch / (num_steps - 1 + 1e-12)
        entropy_coeff = config.INITIAL_ENTROPY_COEFF * (config.FINAL_ENTROPY_COEFF / config.INITIAL_ENTROPY_COEFF) ** progress
        peakiness_coeff = config.INITIAL_PEAKINESS_COEFF * (config.FINAL_PEAKINESS_COEFF / config.INITIAL_PEAKINESS_COEFF) ** progress

        one_hot_adv_float32 = one_hot_adv.to(dtype=torch.float32)
        log_one_hot = torch.log(one_hot_adv_float32 + 1e-12)
        entropy_term = (-one_hot_adv_float32 * (log_one_hot - 1)).sum() * entropy_coeff
        peakiness_term = -torch.log(torch.max(one_hot_adv, dim=1).values + 1e-12).sum() * peakiness_coeff

        regularized_loss = cross_entropy_loss - entropy_term + peakiness_term

        regularized_loss.backward()
        torch.nn.utils.clip_grad_norm_([one_hot_adv], max_norm=config.GRAD_CLIP_MAX_NORM)
        optimizer.step()
        scheduler.step(continuous_loss)

        if output_freq and epoch % output_freq == 0:
            with torch.no_grad():
                assistant_role_len = manager.get_assistant_role_slice().stop - manager.get_adv_slice().stop
                gen_suffix_embeds = suffix_embeds[:, :assistant_role_len, :]
                full_embeds_for_gen = torch.cat([prefix_embeds, adv_embeds, gen_suffix_embeds], dim=1)

                generated = model.generate(
                    inputs_embeds=full_embeds_for_gen,
                    max_length=full_embeds_for_gen.shape[1] + 50,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
                output_str = tokenizer.decode(generated[0][full_embeds_for_gen.shape[1]:], skip_special_tokens=True)
                print(f"\n--- Epoch {epoch} | Generated: ---\n{output_str}\n---\n")

        with torch.no_grad():
            masked_one_hot = get_masked_one_hot_adv(one_hot_adv, non_ascii_toks_tensor)
            adv_token_ids = masked_one_hot.argmax(dim=1)
            discrete_adv_embeds = (F.one_hot(adv_token_ids, embed_weights.shape[0]).to(embed_weights.dtype) @ embed_weights).unsqueeze(0)

            full_embeds_discrete = torch.cat([prefix_embeds, discrete_adv_embeds, suffix_embeds], dim=1)
            logits_discrete = model(inputs_embeds=full_embeds_discrete).logits
            target_logits_disc = logits_discrete[0, loss_slice.start-1:loss_slice.stop-1, :]
            disc_loss = F.cross_entropy(target_logits_disc, target_ids).detach().cpu().item()

            if disc_loss < best_loss:
                best_loss = disc_loss
                best_epoch = epoch
                best_one_hot = masked_one_hot.clone()

    # Pass the manager outward so the final evaluation gets the right template
    return best_one_hot, best_loss, best_epoch, manager

def generate_output(model, tokenizer, manager, adv_one_hot, device, max_length=300):
    """Generate model output using the proper chat template."""
    adv_token_ids = adv_one_hot.argmax(dim=1)
    adv_string = tokenizer.decode(adv_token_ids)

    # FIX: Get correctly templated full string, then clip right before target response
    full_input_ids = manager.get_input_ids(adv_string=adv_string).to(device)
    gen_input_ids = full_input_ids[:manager.get_assistant_role_slice().stop]

    import warnings
    warnings.filterwarnings("ignore", message=".*`do_sample` is set to `False`.*")

    generated_output = model.generate(
        gen_input_ids.unsqueeze(0),
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False
    )

    # Truncate prompt out of the generation output
    input_len = gen_input_ids.shape[0]
    generated_output_string = tokenizer.decode(
        generated_output[0][input_len:].cpu().numpy(),
        skip_special_tokens=True
    ).strip()

    return generated_output_string, adv_token_ids