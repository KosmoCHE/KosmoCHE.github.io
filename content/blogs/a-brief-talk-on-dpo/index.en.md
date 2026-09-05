---
title: 'RL4LLM: 1. A Brief Talk on DPO'
date: 2025-07-12T00:58:24+08:00
draft: false
author: ["Kosmo CHE"]

categories:
    - Large Language Model
tags:
    - Reinforcement Learning
    - Direct Preference Optimization
description: "This blog post notes my understanding of Direct Preference Optimization and the math derivation behind it."
summary: "This blog post notes my understanding of Direct Preference Optimization and the math derivation behind it."
comments: true
images:
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
## Motivation and Overview
Before we start the derivation in detail, we first introduce the motivation and overview of DPO.

In RLHF, we first need to train a reward model to predict the reward of a response given a prompt. Then, we use the reward model to train the policy model under the restriction of the reference model. However, this two-step process is not very efficient. Especially, we at least need to maintain two model weights, the reward model and the reference model,and train one model or maybe two models at the same time. 

To address this issue, DPO proposes a more efficient approach by directly optimizing the policy model in the processing when training the reward model. 

## Derivation
We start with the objective function of the reward model, the most popular choice to model human preferences is Bradley-Terry (BT) model,can be written as:

$$
p^*(y_1 \succ y_2 | x) = \frac{\exp(r^*(x, y_1))}{\exp(r^*(x, y_1)) + \exp(r^*(x, y_2))} \tag{Eq 1}
$$

The reward model is trained to maximize the likelihood of the preference data, which can be written as:

$$
\mathcal{L}_{R}(r_{\phi}, \mathcal{D}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \log \sigma(r_{\phi}(x, y_w) - r_{\phi}(x, y_l)) \tag{Eq 2}
$$

To optimize the policy model in the same time, we need to find a way to convert the reward model into a policy model. In other words, we need to find a way to express the policy model $\pi_{\theta}(y|x)$ in terms of the reward model $r_{\phi}(x, y)$.

All RLHF algorithms can be summarized as the following optimization problem:

$$
\max_{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y|x)} [r_{\phi}(x, y)] - \beta \mathbb{D}_{KL}(\pi_{\theta}(y|x) || \pi_{\text{ref}}(y|x)) \tag{Eq 3}
$$

where $\beta$ is a hyperparameter to control the trade-off between the reward and the KL divergence, $r_{\phi}(x, y)$ is the reward model, and $\pi_{\text{ref}}(y|x)$ is the reference model.

To solve this optimization problem(Eq 3), we can get the optimal policy model $p^*(y|x)$ expressed by the reward model $r_\phi(x, y)$.
$$
\begin{align*}
&\max_{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y|x)} [r_{\phi}(x, y)] - \beta \mathbb{D}_{KL}(\pi_{\theta}(y|x) || \pi_{\text{ref}}(y|x)) \\
& =\max _{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y|x)} [r_{\phi}(x, y) - \beta \log \frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x)}] \\
& = \min_{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y|x)} [\log \frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x)} - \frac{1}{\beta}r_{\phi}(x, y)] \\
&= \min_{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y|x)} [\log \frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x) \exp \left (\frac{1}{\beta}r_{\phi}(x, y)\right)}] \tag{Eq 4}
\end{align*}
$$

Now, we find the **Eq 4 like a KL divergence term which is easy for us to optimize**, but with a non-probability distribution in the denominator. So we need to convert the non-probability distribution into a probability distribution by add a normalization term $Z(x)$, which is the partition function:

$$
Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta}r_{\phi}(x, y)\right) \tag{Eq 5}
$$

so that we can rewrite the denominator as:
$$
\begin{align*}
\pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta}r_{\phi}(x, y)\right) &= \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta}r_{\phi}(x, y)\right) Z(x)^{-1} Z(x) \\
&= \pi^*(y|x) Z(x) \tag{Eq 6}
\end{align*}
$$

where $\pi^*(y|x) = \frac {\pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta}r_{\phi}(x, y)\right)}{\sum_y \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta}r_{\phi}(x, y)\right)}$, means the probability distribution of the reference model after being re-weighted by the reward function.

we substitute Eq 6 into Eq 4:

$$
\begin{align*}
&\min_{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y|x)} [\log \frac{\pi_{\theta}(y|x)}{\pi^*(y|x)} - \log Z(x)]\\
&= \min_{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}} [\mathbb{D}_{KL}(\pi_{\theta}(y|x) || \pi^*(y|x)) - \log Z(x)] \tag{Eq 7}
\end{align*}
$$

Because the second term is a constant, we can ignore it, and the KL-divergence is minimized at 0 when $\pi_{\theta}(y|x) = \pi^*(y|x)$, the Explicit Solution of the optimal policy model is:

$$
\pi_{\theta}(y|x)=\pi^*(y|x) = \frac {\pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta}r_{\phi}(x, y)\right)}{Z(x)} \tag{Eq 8}
$$

However, we can not directly use this solution to train the policy model, because computing the partition function $Z(x)$ is expensive.

According to Eq 8, We can get:
$$
r_{\phi}(x, y) = \beta \log \frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x) \tag{Eq 9}
$$

Finally, we substitute the Eq 9 into the loss function for reward model Eq 2, we can get the loss function for the policy model:
$$
\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_{\theta}(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \tag{Eq 10}
$$


## Reference

1. Rafailov, Rafael, et al. "Direct preference optimization: Your language model is secretly a reward model." Advances in neural information processing systems 36 (2023): 53728-53741.
2. 猛猿. {{<href text="人人都能看懂的DPO数学原理" url="https://zhuanlan.zhihu.com/p/721073733">}}.
3. SmallerFL. {{<href text="Direct Preference Optimization (DPO) 原理详解及公式推导" url="https://zhuanlan.zhihu.com/p/779691018">}}.


