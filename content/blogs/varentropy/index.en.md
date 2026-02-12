---
title: 'Varentropy'
date: 2025-10-22T23:21:10+08:00
draft: true
author: ["Kosmo CHE"]
categories:
    - xxx
tags:
    - xxx
description: "This blog post discusses xxx"
comments: true
images:
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
## The Mathematical Properties of Varentropy

Now we discuss some mathematical properties of varentropy at a discrete distribution with a fixed entropy.

Let $X$ be a discrete random variable with support $\{x_1, x_2, \ldots, x_k\}$ and corresponding probabilities $P(X = x_i) = p_i$, and let $H(X) = H_0$ be the entropy of $X$. The varentropy $V(X)$ can be expressed as:
$$
V(X) = \sum_{i=1}^{V} p_i \left( -\log(p_i) \right)^2 - H_0^2
$$

## Lagrange Multiplier Method
To find the lower bound of varentropy for a fixed entropy $H_0$, we can use the method of Lagrange multipliers. We want to maximize $V(X)$ subject to the constraints that the probabilities sum to 1 and the entropy is fixed.
We can consider it as the following optimization problem:

$$\begin{align}
&\min V(X) = \sum_{i=1}^{k} p_i \left( -\log p_i \right)^2 - H_0^2 \\
&\text{s.t. }  \sum_{i=1}^{k} p_i = 1,\  -\sum_{i=1}^{k} p_i \log(p_i) = H_0
\end{align}$$

We can set up the Lagrangian function:
$$\mathcal{L}(p_1, p_2, \ldots, p_k, \lambda, \mu) = V(X) + \lambda \left( \sum_{i=1}^{k} p_i - 1 \right) + \mu \left( -\sum_{i=1}^{k} p_i \log(p_i) - H_0 \right)$$
Taking the partial derivatives of $\mathcal{L}$ with respect to $p_i$, $\lambda$, and $\mu$, and setting them to zero gives us the following system of equations:

$$\begin{align}
\frac{\partial \mathcal{L}}{\partial p_i} &= \left( -\log(p_i) \right)^2 + 2 \log(p_i) + \lambda + \mu ( -\log(p_i) - 1 ) = 0, \quad \text{for } i = 1, 2, \ldots, k \\
\frac{\partial \mathcal{L}}{\partial \lambda} &= \sum_{i=1}^{k} p_i - 1 = 0 \\
\frac{\partial \mathcal{L}}{\partial \mu} &= -\sum_{i=1}^{k} p_i \log(p_i) - H_0 = 0
\end{align}$$

Simplifying the first equation, we have:
$$\left(\log p_i \right)^2 + (2 - \mu) \log p_i + (\lambda - \mu) = 0$$
This is a quadratic equation about $t =\log p_i$:

$$
t^2 + (2 - \mu) t + (\lambda - \mu) = 0
$$

Given the fixed $\lambda$ and $\mu$, this equation has at most two solutions for $t$, denoted as $t_1$ and $t_2$. Therefore, the probabilities $p_i$ can take at most two distinct values.

**Conclusion 1 (Two-Level Structure):** Any extremal solution takes on at most two distinct probability values. That is, the optimal distribution can be described by a "two-level" structure: the probabilities $p$ take on only two values, $p_a$ and $p_b$, occurring at $M$ and $N$ points, respectively.

## Two-Level Structure
Based on Conclusion 1, we can express the probabilities as follows:

- Let the first level have size $M$ and a total probability mass of $\alpha$. The probability for each point in this level is then $p_a = \alpha/M$, and the Information Content is $I_a = -\log(p_a) = \log(M/\alpha)$.
- The second level has size $N$ and a total probability mass of $1 - \alpha$. Consequently, the probability for each point in this level is $p_b = (1 - \alpha)/N$, and the Information Content is $I_b = -\log(p_b) = \log(N/(1 - \alpha))$.

Thus, we can express the entropy and varentropy in terms of $M$, $N$, and $\alpha$:

$$\begin{align}
H(\alpha, M, N) &= -M \cdot p_a \log p_a - N \cdot p_b \log p_b \\
&= -\alpha \log(\frac{\alpha}{M}) - (1 - \alpha) \log(\frac{1 - \alpha}{N})\\
&= \alpha \log M + (1 - \alpha) \log N + H_b(\alpha)
\end{align} \tag{1}$$

Here, $H_b(\alpha) = -\alpha \log(\alpha) - (1 - \alpha) \log(1 - \alpha)$ is the binary entropy function.

$$\begin{align}
V(\alpha, M, N) &= M \cdot p_a \left( -\log(p_a) \right)^2 + N \cdot p_b \left( -\log(p_b) \right)^2 - H(X)^2 \\
&= \alpha \left( \log(\frac{M}{\alpha}) \right)^2 + (1 - \alpha) \left( \log(\frac{N}{1 - \alpha}) \right)^2 - H(X)^2\\
&= \alpha (1 - \alpha) \left[ \log(\frac{1- \alpha}{\alpha}) - \log(\frac{N}{M}) \right]^2 \\
\end{align} \tag{2}$$

Thus, the optimization problem reduces to finding the optimal values of $M$, $N$, and $\alpha$ that minimize $V(X)$ while satisfying the entropy constraint $H(X) = H_0$.

Varentropy $V(\alpha, M, N)$ is non-negative, so the infimum of varentropy is 0. But when can varentropy reach 0? We analyze it from Special Case and General Case.

## Special Case: Varentropy Collapse
For $\alpha \in [0, 1]$, $\alpha (1 - \alpha) \ge 0$, and the square term is always non-negative. To make $V(\alpha, M, N) = 0$, we need achieve one of the following conditions:

1. $\alpha \to 0$ 
2. $\alpha \to 1$
3. $\log(\frac{1- \alpha}{\alpha}) - \log(\frac{N}{M}) = 0$

With the Eq 1, we can analyze these three conditions:
1. If $\alpha \to 1$, the probability mass is concentrated on the first level and uniformly distributed among the $M$ points. The entropy becomes:
   $$H(1, M, N) = \log(M)$$
2. If $\alpha \to 0$, the probability mass is concentrated on the second level and uniformly distributed among the $N$ points. The entropy becomes:
   $$H(0, M, N) = \log(N)$$

3. If $\log(\frac{1- \alpha}{\alpha}) - \log(\frac{N}{M}) = 0$, we have $\frac{1- \alpha}{\alpha} = \frac{N}{M}$, which leads to $\alpha = \frac{M}{M + N}$.The probability mass is distributed uniformly among the $M$ and $N$ points. Substituting this into the entropy expression gives:
   $$\begin{align}
   H(\frac{M}{M + N}, M, N) &= \frac{M}{M + N} \log(M)+ \frac{N}{M + N} \log(N) + H_b(\frac{M}{M + N}) \\
   &= \log(M + N)
   \end{align}$$

$M=0$ or $N=0$ can be treated as the first two conditions.

The three conditions indicate that the varentropy reaches its minimum value of 0 when the two-level structure collapses into a single-level uniform distribution over either $M$, $N$, or $M + N$ points.

**Conclusion 2 (Varentropy Collapse):** $V(X) = 0$ if and only if X is a uniform distribution (i.e., $p_i = 1/k$ for $k$ support points), in which case the entropy $H_0 = \log(k)$.

## General Case: Varentropy Lower Bound
Then we discuss the more general case when distribution is not uniform, i.e., $H_0 \neq \log(k)$.

For any given entropy $H_0 > 0$ (such that $H_0 \neq \log k$), there must exist a unique positive integer $k_0$ such that $H_0$ lies between two adjacent uniform distribution entropies:

$$
\log(k_0) < H_0 < \log(k_0 + 1)
$$

According to **Conclusion 1**, the extremal solution has a two-level structure $(M, N, \alpha)$.So we have:
$$M + N \ge k_0 + 1$$
and
$$\min(M, N) \le k_0$$


To simplify the analysis, we can assume $M \le N$, so we have:

$$M \in [1, k_0], M \in \mathbb{Z}^+$$

$$
N \in [\max(M, k_0+1-M), \infty), N \in \mathbb{Z}^+
$$

Consider the fixed $M$ and $H_0$, $\alpha$ is an implicit function of $N$ through the entropy constraint Eq 1. We can analyze how varentropy $V(\alpha, M, N)$ changes with respect to $N$ by calculating the derivative $\frac{dV}{dN}$.

The appendix provides a detailed derivation of $\frac{dV}{dN}$, and the result is:

$$\frac{d\alpha}{dN} = - \frac{\partial H / \partial N}{\partial H / \partial \alpha} = - \frac{(1-\alpha)/N}{\Delta}$$

$$\frac{dV}{dN} = \frac{1-\alpha}{N} (2 - \Delta)$$

Where $\Delta = \log(\frac{(1-\alpha) M}{\alpha N}) = I_a - I_b$.

$$\frac{dV}{dN} = \frac{1-\alpha}{N} (2 - (I_a - I_b))$$

Here, We Fix $N$, entropy $H(\alpha)= \alpha \log M + (1-\alpha) \log N + H_b(\alpha)$ is a function of $\alpha$, and its derivative with respect to $\alpha$ is given by:

$$\frac{dH}{d\alpha} = \log(\frac{(1-\alpha) M}{\alpha N}) = \Delta$$

It is a curve that increases from $H(0) = \log(N)$ to $H(\frac{M}{M+N}) = \log(M+N)$, and then decreases to $H(1) = \log(M)$.

Equation $H(\alpha) = H_0$ has two solutions for $\alpha$ when $H_0 \in [\log N, \log (M+N)]$, and one solution when $H_0 \in [\log M, \log N)$.

We denote the two solution branches as $\alpha_1$ and $\alpha_2$:
- $\alpha_1 \in (0, \frac{M}{M+N})$ when $H_0 \in [\log M, \log N]$. In this branch, $I_a - I_b > 0$.
- $\alpha_2 \in (\frac{M}{M+N}, 1)$. In this branch, $I_a - I_b < 0$.

**Discussing Branch 2**

In Branch 2, since $I_a - I_b < 0$, we have $\frac{dV}{dN} > 0$. This means that as $N$ increases, $V(\alpha_2, M, N)$ also increases. Therefore, for a fixed $M$, the minimum varentropy occurs at the smallest possible value of $N$, which is $N = \max(M, k_0 + 1 - M)$.

$$
V_\min = \min_{M \in [1, k_0]} V(\alpha_2(M), M, \max(M, k_0 + 1 - M))
$$

**Discussing Branch 1**
$M \in [1, k_0], N \in [\max(M, k_0 + 1 - M), k_0)$

## Appendix1: Detailed Derivation of $\frac{dV}{dN}$
Because $H_0$ is constant, $\alpha$ is an implicit function of $N$. To calculate $\frac{dV}{dN}$, we use the chain rule:
$$\frac{dV}{dN} = \frac{\partial V}{\partial \alpha} \frac{d\alpha}{dN} + \frac{\partial V}{\partial N}$$

We need to calculate $\frac{d\alpha}{dN}$, $\frac{\partial V}{\partial \alpha}$, and $\frac{\partial V}{\partial N}$ step by step.

Step 1: Calculate $\frac{d\alpha}{dN}$
We take the total derivative of the constraint equation $H(\alpha, N) = H_0$ both sides with respect to $N$:
$$\frac{d(H_0)}{dN} = 0$$

$$\frac{d(H(\alpha, N))}{dN} = \frac{\partial H}{\partial \alpha} \frac{d\alpha}{dN} + \frac{\partial H}{\partial N} \frac{dN}{dN} = 0$$

$$\frac{\partial H}{\partial \alpha} \frac{d\alpha}{dN} + \frac{\partial H}{\partial N} = 0$$$$\implies \frac{d\alpha}{dN} = - \frac{\partial H / \partial N}{\partial H / \partial \alpha}$$

Step 1a: Calculate the denominator $\frac{\partial H}{\partial \alpha}$:

$$H = [-\alpha \log \alpha - (1-\alpha) \log(1-\alpha)] + \alpha \log M + (1-\alpha) \log N$$

$$\frac{\partial H}{\partial \alpha} = \frac{\partial}{\partial \alpha}[-\alpha \log \alpha] + \frac{\partial}{\partial \alpha}[-(1-\alpha) \log(1-\alpha)] + \frac{\partial}{\partial \alpha}[\alpha \log M] + \frac{\partial}{\partial \alpha}[(1-\alpha) \log N]$$

- $\frac{\partial}{\partial \alpha}[-\alpha \log \alpha] = -[1 \cdot \log \alpha + \alpha \cdot \frac{1}{\alpha}] = -\log \alpha - 1$

- $\frac{\partial}{\partial \alpha}[-(1-\alpha) \log(1-\alpha)] = -[(-1) \cdot \log(1-\alpha) + (1-\alpha) \cdot \frac{1}{1-\alpha} \cdot (-1)] $$= -[-\log(1-\alpha) - 1] = \log(1-\alpha) + 1$

- $\frac{\partial}{\partial \alpha}[\alpha \log M] = \log M$
- $\frac{\partial}{\partial \alpha}[(1-\alpha) \log N] = -\log N$

Combining these results, we have:
$$\frac{\partial H}{\partial \alpha} = -\log \alpha - 1 + \log(1-\alpha) + 1 + \log M - \log N = \log(\frac{(1-\alpha) M}{\alpha N})$$

We denote $\log(\frac{(1-\alpha) M}{\alpha N})$ as $\Delta$ for simplicity.

Step 1b: Calculate the numerator $\frac{\partial H}{\partial N}$:

$$\frac{\partial H}{\partial N} = \frac{\partial}{\partial N} [(1-\alpha) \log N] = (1-\alpha) \cdot \frac{1}{N} = \frac{1-\alpha}{N}$$

Step 1c: Combine the results to get $\frac{d\alpha}{dN}$:

$$\frac{d\alpha}{dN} = - \frac{\partial H / \partial N}{\partial H / \partial \alpha} = - \frac{(1-\alpha)/N}{\Delta} = - \frac{1-\alpha}{N \Delta}$$

Step 2: Calculate Partial Derivatives of V

Step 2a: Calculate $\frac{\partial V}{\partial \alpha}$

$$V = \alpha(1-\alpha) \Delta^2 = (\alpha - \alpha^2) \Delta^2$$

$$\frac{\partial V}{\partial \alpha} = \frac{\partial(\alpha - \alpha^2)}{\partial \alpha} \cdot \Delta^2 + (\alpha - \alpha^2) \cdot \frac{\partial(\Delta^2)}{\partial \alpha}$$

$$\frac{\partial V}{\partial \alpha} = (1 - 2\alpha) \Delta^2 + \alpha(1-\alpha) \cdot \left( 2\Delta \cdot \frac{\partial \Delta}{\partial \alpha} \right)$$

To calculate $\frac{\partial \Delta}{\partial \alpha}$, we have:

$$\Delta = \log M - \log N + \log(1-\alpha) - \log \alpha$$

$$\frac{\partial \Delta}{\partial \alpha} = - \frac{1}{\alpha(1-\alpha)}$$

Substituting this back into the expression for $\frac{\partial V}{\partial \alpha}$ gives:

$$\frac{\partial V}{\partial \alpha} = (1 - 2\alpha) \Delta^2 + \alpha(1-\alpha) \cdot 2\Delta \cdot \left( - \frac{1}{\alpha(1-\alpha)} \right)$$

$$\frac{\partial V}{\partial \alpha} = (1 - 2\alpha) \Delta^2 - 2\Delta$$

Step 2b: Calculate $\frac{\partial V}{\partial N}$

$$V = \alpha(1-\alpha) \Delta^2$$

$V$ depends on $N$ only through $\Delta$, so we have:

$$\frac{\partial V}{\partial N} = \alpha(1-\alpha) \cdot \frac{\partial(\Delta^2)}{\partial N} = \alpha(1-\alpha) \cdot \left( 2\Delta \cdot \frac{\partial \Delta}{\partial N} \right)$$

Calculating $\frac{\partial \Delta}{\partial N}$:

$$\Delta = \log M - \log N + \log(1-\alpha) - \log \alpha$$

$$\frac{\partial \Delta}{\partial N} = - \frac{1}{N}$$

Substituting this back into the expression for $\frac{\partial V}{\partial N}$ gives:

$$\frac{\partial V}{\partial N} = \alpha(1-\alpha) \cdot 2\Delta \cdot \left( - \frac{1}{N} \right) = - \frac{2\alpha(1-\alpha)\Delta}{N}$$

Step 3: Combine All Results to Get $\frac{dV}{dN}$

Chain Rule:

$$\frac{dV}{dN} = \frac{\partial V}{\partial \alpha} \frac{d\alpha}{dN} + \frac{\partial V}{\partial N}$$

Substituting the expressions derived in Steps 1 and 2:

$$\frac{dV}{dN} = \underbrace{\left[ (1 - 2\alpha) \Delta^2 - 2\Delta \right]}_{\partial V / \partial \alpha} \cdot \underbrace{\left( - \frac{1-\alpha}{N \Delta} \right)}_{d\alpha / dN} + \underbrace{\left( - \frac{2\alpha(1-\alpha)\Delta}{N} \right)}_{\partial V / \partial N}$$

After simplification, we have:

$$\frac{dV}{dN} = \left[ - (1 - 2\alpha) \Delta \left( \frac{1-\alpha}{N} \right) + 2 \left( \frac{1-\alpha}{N} \right) \right] - \frac{2\alpha(1-\alpha)\Delta}{N}$$

Extracting the common factor $\frac{1-\alpha}{N}$, we get:

$$\frac{dV}{dN} = \left( \frac{1-\alpha}{N} \right) \cdot \left[ - (1 - 2\alpha) \Delta + 2 - 2\alpha\Delta \right]$$

Expanding the terms inside the brackets:
$$\frac{dV}{dN} = \left( \frac{1-\alpha}{N} \right) \cdot \left[ - \Delta + 2\alpha\Delta + 2 - 2\alpha\Delta \right] = \left( \frac{1-\alpha}{N} \right) \cdot (2 - \Delta)$$