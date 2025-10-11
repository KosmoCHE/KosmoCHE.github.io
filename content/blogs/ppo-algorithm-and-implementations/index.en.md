---
title: 'RL4LLM: 2. PPO Algorithm and Implementation Details'
date: 2025-07-21T19:58:49+08:00
draft: false
author: ["Kosmo CHE"]
categories:
    - Large Language Model
tags:
    - Reinforcement Learning
    - Proximal Policy Optimization
description: "This blog post introduces RLHF-PPO algorithm with code implementation."
comments: true
images:
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
## 1. Overview of PPO Implementation
The simplified code for the RLHF with PPO algorithm in **TRL** is as follows:
```python {title="The simplified code of PPO"}
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
        # cal advantages
        for t in reversed(range(gen_length)):
            nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
            delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
            lastgaelam = delta + args.gamma * args.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
    # Train
    for ppo_epoch_idx in range(num_ppo_epochs):
        logprobs_new_policy, v_pred = policy_model(batch)
        # compute value_loss
        v_pred_clipped = clip(v_pred, 
                            values - cliprange_value, 
                            values + cliprange_value)
        vf_loss1 = (v_pred - returns) ** 2
        vf_loss2 = (v_pred_clipped - returns) ** 2
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
## 2. Theory
To express the PPO algorithm explicitly, I will ignore the **kl-penalty term** and **clip operation** for simplicity.


### 2.1. Policy Loss
#### 2.1.1 Policy Gradient

The optimization objective of reinforcement learning is to maximize the expected return:
$$\max_{\pi_\theta}\mathcal{J}(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]= \sum_\tau R(\tau)P(\tau|\pi_\theta) \tag{Eq 1}$$

where $\tau$ is a trajectory sampled from the policy $\pi_\theta$, and $R(\tau)$ is the return of the trajectory.

Then the policy gradient can be derived as follows(details can be found in the [[1]](https://zhuanlan.zhihu.com/p/7461863937)):
$$
\begin{align*}
\nabla_{\theta}\mathcal{J}(\pi_\theta) &= \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)\nabla log(P(\tau|\pi_\theta))] \\
& = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau) \sum_{t=0}^{T_n-1}\nabla log(\pi_\theta(a_t|s_t))] \tag{Eq 2}
\end{align*}
$$
where $T_n$ is the length of the trajectory, and $s$ is the state at time step $t$.
#### 2.1.2. Improvement of  $R(\tau)$ - GAE
As we consider Eq 2, the return $R(\tau)$ is a sum of rewards on the trajectory, but the $\pi_\theta(a|s)$ is a probability distribution on single step. We may think this is not a good choice to use the sum of rewards as the return. In fact there are many ways to improve the return $R(\tau)$, but the most common one is Generalized Advantage Estimation (GAE).

$$
\begin{align*}
&\delta_\phi(s_t,a_t) = r_t +\gamma V_\phi(s_{t+1}) - V_\phi(s_t) \\
&A_\phi(s_t,a_t)=\sum_{l=0}^{\infty}(\gamma \lambda)^l \delta_\phi(s_{t+l},a_{t+l}) \tag{Eq 3}
\end{align*}
$$

where $A_\phi(s_t,a_t)$ is the advantage function, $r_t$ is the reward at time step $t$, $\gamma$ is the discount factor, and $V_\phi(s)$ is the value function.

The new policy gradient can be written as:
$$
\nabla_{\theta}\mathcal{J}(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=0}^{T_n-1} A_\phi(s_t,a_t)\nabla log(\pi_\theta(a_t|s_t))] \tag{Eq 4}
$$

According to the gradient Eq 4, we can get the new optimization objective:
$$
\begin{align*}
\max_{\pi_\theta}\mathcal{J}(\pi_\theta) &= \mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=0}^{T_n-1} A_\phi(s_t,a_t) log(\pi_\theta(a_t|s_t))] \\
& \approx \frac{1}{N}\sum_{n=1}^{N}\sum_{t=0}^{T_n-1} A_\phi(s_t,a_t) log(\pi_\theta(a_t|s_t)) \\
\Leftrightarrow \max_{\pi_\theta}\mathcal{J}(\pi_\theta) & = \frac{1}{\sum_{n=1}^{N} T_n} \sum_{n=1}^{N}\sum_{t=0}^{T_n-1}A_\phi(s_t,a_t) log(\pi_\theta(a_t|s_t)) \\
&= \mathbb{E}_t[A_\phi(s_t,a_t) log(\pi_\theta(a_t|s_t))]  \tag{Eq 5}
\end{align*}
$$

Because the solution is equal between the expectation of the trajectory $\tau$ and the expectation of the time step $t$, we can derive line 2 to line3 in Eq 5.


#### 2.1.3. Importance Sampling
In the above equation, we can see that the policy gradient is computed by the advantage function and the log probability of the action taken by the policy. However, in practice, we often have a **old policy** $\pi_{\text{old}}(a|s)$, which is used to generate the trajectory $\tau$. We will use these trajectories to update the policy $\pi_\theta(a|s)$ for `ppo_epochs` times.

The Policy Gradient can be rewritten as:
$$
\begin{align*}
\nabla_{\theta}\mathcal{J}(\pi_\theta) &= \mathbb{E}_t[A_\phi(s_t,a_t)\nabla log(\pi_\theta(a_t|s_t))]\\
&=\mathop{\mathbb{E}_t}\limits_{\tau \sim \pi_{\text{old}}}[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)} A_\phi(s_t,a_t)\nabla log(\pi_\theta(a_t|s_t)) ] \tag{Eq 6}
\end{align*}
$$

At last, we can get the final optimization objective:
$$
\mathcal{J}(\pi_\theta) = \mathop{\mathbb{E}_t}\limits_{\tau \sim \pi_{\text{old}}}[A_\phi(s_t,a_t) \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}] \tag{Eq 7}
$$

### 2.2 Critic Loss
$$
\mathcal{L}_{\text{critic}} = \mathbb{E}_t[(V_\phi(s_t) - A_\phi(s_t,a_t)-V_{\text{old}}(s_t))^2] \tag{Eq 8}
$$
## References
1. 猛猿. {{<href text="人人都能看懂的RL-PPO理论知识" url="https://zhuanlan.zhihu.com/p/7461863937">}}.
2. TRL: {{<href text="PPO Trainer Implementation" url="https://github.com/huggingface/trl/blob/44e6c153a517ebe6da572ad3c882cbe1e90629b6/trl/trainer/ppo_trainer.py#L412">}}.