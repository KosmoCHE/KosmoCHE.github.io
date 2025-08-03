---
title: 'RL4LLM: 2. PPO Algorithm and Implementation Details'
date: 2025-07-21T19:58:49+08:00
draft: true
author: ["Kosmo CHE"]
keywords: 
    - LLM
    - Reinforcement Learning
    - PPO
categories:
    - RL4LLM
tags:
    - PPO
description: "This blog post introduces RLHF-PPO algorithm with code implementation."
summary: "This blog post introduces RLHF-PPO algorithm with code implementation."
comments: true
images:
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
## Overview
The pseudo-code for the RLHF with PPO algorithm in **TRL** is as follows:
```python

for batch in dataloader:
    # Sample
    with torch.no_grad():
        old_policy_outputs,logprobs_old_policy = policy_model(batch)
        logprobs_reference = reference_model(batch)
        scores = reward_model(query+old_policy_outputs)
        values = value_model(query+old_policy_outputs)

        kl = logprobs_old_policy - logprobs_reference
        # reward function have different implementation methods
        rewards = scores - kl * kl_coef 
        lastgaelam = 0
        advantages_reversed = []
        gen_length = responses.shape[1]
        for t in reversed(range(gen_length)):
            nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
            delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
            lastgaelam = delta + args.gamma * args.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
    # Train
    for ppo_epoch_idx in range(num_ppo_epochs):
        logprobs_new_policy, v_pred = policy_model(batch)
        # compute vf_loss
        v_pred_clipped = clip(v_pred, 
                            values - cliprange_value, 
                            values + cliprange_value)
        vf_loss1 = (v_pred - rewards) ** 2
        vf_loss2 = (v_pred_clipped - rewards) ** 2
        vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
        # compute policy_loss
        important_ratio = logprobs_new_policy - logprobs_old_policy
        policy_loss1 = -(exp(important_ratio) * advantages)
        policy_loss2 = -(clip(exp(important_ratio), 
                            1.0 - cliprange_ratio, 
                            1.0 + cliprange_ratio) \ 
                        * advantages)
        policy_loss = torch.max(policy_loss1, policy_loss2).mean()
        loss = policy_loss + vf_loss * args.vf_coef
```
## Implementation Details
