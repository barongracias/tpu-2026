# GRPO and Related RL Methods for GSM8K with Small LLMs

## Overview

This document summarizes prior work on applying Group Relative Policy Optimization (GRPO) and related REINFORCE-style algorithms to math reasoning benchmarks such as GSM8K, focusing on small to mid‑sized language models (≈0.5–7B parameters). It consolidates what tends to work and not work in practice, highlights key theoretical and empirical findings, and lists the most relevant research papers and practitioner resources for coursework‑level experiments on models like Gemma‑3‑1B‑IT.[1][2][3][4]

The emphasis is on:

- Response-level RL with verifiable rewards (RLVR/GRPO/RLOO) on GSM8K.
- Small-group policy-gradient estimators (group size K typically between 2 and 8).
- Behavior of standard GRPO versus leave‑one‑out (RLOO) style estimators.


## Core Algorithms and Setting

### GRPO and RLVR

Group Relative Policy Optimization (GRPO) is a critic‑free policy-gradient method that uses group‑normalized advantages over multiple responses sampled per prompt.  For each prompt, K responses are drawn from the current policy, rewards are evaluated (often via verifiable rules such as numeric correctness on GSM8K), and an advantage is computed by subtracting the group mean and normalizing by group standard deviation.[4][5][6]

Reinforcement Learning with Verifiable Rewards (RLVR) is a broader framework in which rewards are computed by deterministic verifiers rather than learned reward models; GRPO is one of the main algorithmic choices within RLVR for LLM post‑training.[7][8]

### REINFORCE and RLOO

REINFORCE is the classic vanilla policy-gradient algorithm that optimizes the expected reward of sampled trajectories via log‑probability gradients multiplied by scalar advantages.  Variants differ mainly in how they construct baselines and advantages.[9][10]

REINFORCE Leave‑One‑Out (RLOO) constructs an unbiased per‑sample advantage by subtracting, for each response in a group, the mean reward of the other K−1 responses for the *same* prompt.  For group size K=2, this reduces to each sample using the other sample’s reward as its baseline, so advantages become pairwise reward differences.[11][10]


## What Tends to Work Well

### 1. Verifiable, High‑Signal Rewards on GSM8K

GSM8K is well‑suited to RLVR because each problem has a single numeric answer that can be extracted and compared automatically, enabling deterministic, high‑signal rewards.  Successful systems combine:[2][12]

- A strict correctness reward (often Bernoulli: 1 for exact final numeric match, 0 otherwise).
- Auxiliary shaping terms for formatting (e.g., bonus if the answer appears in a prescribed pattern) and sometimes for brevity or coherence.[2][7]

This design reduces reliance on learned reward models and makes GRPO/RLOO training more stable and interpretable, as reward noise comes mainly from sampling and parsing, not from model‑based reward estimation.[7][2]

### 2. Focusing GRPO/RLVR on Hard Examples

"Hard Examples Are All You Need" studies GRPO post‑training under annotation budgets and systematically compares training on easy, medium, hard, or random subsets of tasks.  For math reasoning benchmarks like GSM8K, it finds that training GRPO on the hardest examples (defined by base model difficulty) yields significantly larger performance gains than training on easy or random subsets, with reported improvements of tens of percentage points in some configurations.[13][4]

Intuitively, hard examples generate more diverse rewards within each group, leading to informative, high‑variance advantages; easy examples tend to produce saturated rewards that carry little learning signal after group normalization.  This suggests that for small models, targeting GRPO/RLVR on mis‑solved or high‑uncertainty GSM8K problems is more effective than uniform sampling.[13]

### 3. GRPO‑Style Algorithms on Small Models

Public RLVR/GRPO experiments on Qwen2.5‑Math‑1.5B and 0.5B demonstrate that GRPO can substantially improve GSM8K accuracy even for relatively small models, provided that verifiable rewards and conservative hyperparameters are used.  For example, Qwen2.5‑Math‑1.5B improves from around the low‑70s to high‑70s GSM8K accuracy after RLVR+GRPO, while Qwen2.5‑0.5B sees double‑digit point gains.[2]

VERL’s GRPO‑LoRA baselines further show that applying GRPO with LoRA adapters to Qwen2.5‑0.5B and 1.5B Instruct models yields several point improvements over the original checkpoints on GSM8K.  These results indicate that GRPO is practical at 0.5–1.5B scale, but also highlight sensitivity to reward design and hyperparameters.[3]

### 4. Variance‑Reduced REINFORCE/RLOO for RLHF

"Back to Basics" revisits REINFORCE‑style optimization for RLHF and shows that properly tuned REINFORCE and RLOO can match or outperform PPO‑style methods on several alignment tasks, while being simpler to implement and more computationally efficient.  It emphasizes the role of good baselines (including leave‑one‑out) in reducing variance and improving sample efficiency.[10][14]

Follow‑up work applying RLOO to small LLMs for instruction following and math reasoning reports that RLOO with a reward model or verifiable rewards can outperform SFT and DPO baselines in alignment and task performance, particularly when combined with techniques like synthetic data augmentation and best‑of‑N sampling.  This supports using RLOO‑type estimators as strong baselines or variants next to GRPO in coursework experiments.[15]

### 5. GRPO Variants for Math Reasoning (λ‑GRPO, GRPO‑MA)

Several recent papers propose modifications to GRPO specifically aimed at multi‑step reasoning tasks:

- **λ‑GRPO** introduces a learnable token‑level weighting parameter that addresses length bias and token credit assignment; it achieves consistent 1–2 point improvements over vanilla GRPO on mathematical benchmarks for Qwen2.5 models (1.5B–7B) without extra compute or data.[16][17]
- **GRPO‑MA** (multi‑answer GRPO) generates multiple answers per thought trace and constructs advantages in a way that decouples gradients between the reasoning chain and final answer, reducing variance and improving training stability. It consistently outperforms GRPO on GSM8K and HumanEval under matched compute budgets.[18]

Although these variants go beyond basic GRPO, their positive results reinforce core takeaways: careful token‑level credit assignment and control of length bias are important for math reasoning RL on small models.[16][18]


## What Tends Not to Work or Fails Frequently

### 1. Naïve GRPO on Very Small Models without Careful Tuning

Practitioner reports applying GRPO to very small models (e.g., 0.5B parameter Qwen) describe collapse or degenerate behavior, such as models that output only newlines or trivial responses despite seemingly healthy reward curves.  These failures are attributed to overly aggressive learning rates, weak KL regularization, and reward designs that inadvertently encourage pathological outputs.[19][4]

Empirical evidence suggests that models below roughly 3B parameters are more fragile under GRPO updates, requiring conservative hyperparameters, strong KL penalties, and close monitoring of qualitative behavior to avoid collapse.[19][3]

### 2. SFT‑Only Approaches on GSM8K

RLVR/GRPO experiments show that simply applying supervised fine‑tuning on GSM8K solutions (i.e., sequence‑level MLE on CoT) can sometimes *degrade* few‑shot performance compared to the base model, especially when evaluated under different prompting conditions.  In contrast, RLVR+GRPO using verifiable rewards consistently improves accuracy under the same evaluation setup.[2]

This highlights that SFT alone is not a perfect surrogate for outcome‑aligned RL, particularly on GSM8K where the final numeric answer is what matters—the mismatch between training objective (log‑likelihood of teacher solutions) and evaluation metric (correctness of model‑generated solutions) can lead to regressions.[2]

### 3. Uniform Sampling over Mostly Easy GSM8K Examples

The "Hard Examples" study finds that training GRPO on easy or randomly sampled examples is substantially less effective than training exclusively on hard examples.  In particular, when many examples are easy for the current model, group rewards become similar within each prompt, and GRPO’s group‑normalized advantages approach zero, yielding almost no learning signal.[13]

Thus, uniform sampling over GSM8K (which contains many relatively simple problems) can waste RL budget and produce limited gains, especially once the base model reaches moderate accuracy.[4]

### 4. Ignoring Length Bias and Token-Level Credit Assignment

Vanilla GRPO assigns the same scalar advantage to all tokens in a completion, introducing length bias (longer completions receive more total gradient) and coarse credit assignment (early reasoning steps are penalized or rewarded identically to late steps regardless of their contribution).[18][16]

Recent work (λ‑GRPO, GRPO‑MA, DAPO, SSVPO) argues that ignoring these issues can limit performance on math and coding tasks and lead to unstable training dynamics, and demonstrates persistent gains when token‑level weighting and step‑level credit assignment are properly handled.[20][21][16][18]

### 5. Over‑Aggressive Hyperparameters (LR, Entropy, Weak KL)

Comparative studies of PPO, GRPO, and RLOO for LLM post‑training (including policy gradient cheat sheets and practitioner notes) emphasize that large learning rates, high entropy bonuses, and weak KL or trust‑region constraints often cause reward hacking and unstable training.[22][23][24]

For GSM8K‑style tasks, successful runs on small models typically use: small learning rates on LoRA adapters, non‑trivial KL penalties to keep the policy near the reference, and limited entropy bonuses.  These choices reduce gradient variance and help align improvements in reward with improvements in evaluation accuracy.[3][2]


## Key High-Level Takeaways for GSM8K + GRPO/RLOO on Small LLMs

1. **Verifiable rewards are central.** Deterministic correctness checks on GSM8K (plus simple formatting rewards) dramatically simplify RLHF and make GRPO/RLOO practical and interpretable for small models.[12][2]
2. **Data selection matters as much as algorithm choice.** Training GRPO on a hard subset of GSM8K, rather than uniformly on all problems, yields substantially larger gains.[4][13]
3. **GRPO works at small scales but is fragile.** GRPO and GRPO‑like methods can significantly improve GSM8K performance for 0.5–1.5B models when rewards and hyperparameters are tuned carefully; naïve configurations often fail.[3][2]
4. **RLOO is a strong, simple baseline.** REINFORCE with leave‑one‑out baselines (RLOO) is theoretically unbiased and empirically competitive with PPO‑style methods in RLHF, and is particularly attractive for response‑level RL with small group sizes.[11][15][10]
5. **Token-level and group-level design choices have real impact.** Methods that address token‑level credit assignment and group‑level variance (λ‑GRPO, GRPO‑MA, related algorithms) consistently outperform vanilla GRPO on math reasoning benchmarks, pointing to length bias and credit assignment as key levers.[17][16][18]


## Recommended References and URLs

### Scholarly Papers (Peer-Reviewed or arXiv)

| Topic | Reference | Type | URL |
|-------|-----------|------|-----|
| REINFORCE/RLOO for RLHF | Ahmadian et al., "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs" | ACL 2024 | https://arxiv.org/abs/2402.14740 [10] |
| RLOO on small LMs and math | Han & Zhang, "Reinforcement learning fine-tuning of language model for instruction following and math reasoning" | arXiv 2025 | https://arxiv.org/abs/2506.21560 [15] |
| GRPO / Group Robust Preference Optimization | Anonymous, "Group Robust Preference Optimization" | NeurIPS-style | https://proceedings.neurips.cc/paper_files/paper/2024/file/4147dfaa46cd7e20a2aecb91097ae8cc-Paper-Conference.pdf [5] |
| Hard example selection for GRPO | "Hard Examples Are All You Need: Maximizing GRPO Post-Training Under Annotation Budgets" | arXiv / OpenReview 2026 | https://openreview.net/forum?id=d4pQmxezYn [13] |
| λ‑GRPO and token preferences | "Unifying the GRPO Frameworks with Learnable Token Preferences" | arXiv 2025 | https://arxiv.org/abs/2510.06870 [16] |
| GRPO‑MA for multi-answer reasoning | "GRPO-MA: Multi-Answer Generation in GRPO for Stable and Efficient LLM RL" | OpenReview 2026 | https://openreview.net/forum?id=g33DGvnHYd [18] |
| RLVR and Tulu 3 post-training | Ivison et al., "Tulu 3: Pushing Frontiers in Open Language Model Post-Training" | arXiv 2024 | https://arxiv.org/abs/2411.15124 [8] |
| Control variates and leave-one-out baselines | Titsias & Shi, "Double Control Variates for Gradient Estimation in Discrete Latent Variable Models" | PMLR 151 | https://proceedings.mlr.press/v151/titsias22a.html [25] |

### Practitioner Docs, Blogs, and Code

| Resource | Focus | Type | URL |
|----------|-------|------|-----|
| Swift RLOO docs | Formal definition and implementation notes for REINFORCE Leave-One-Out (RLOO) in LLM RL | Library docs | https://swift.readthedocs.io/en/v3.12/Instruction/GRPO/AdvancedResearch/RLOO.html [11] |
| TRL RLOO Trainer | Practical RLOO trainer for Transformers with implementation details | Library docs | https://mintlify.wiki/huggingface/trl/rloo-trainer [26] |
| RLHF Book (policy gradient chapter + cheat sheet) | Equations and implementation tips for PPO, GRPO, RLOO | Online book | https://rlhfbook.com [22][23] |
| GRPO-based post-training summary | Overview of GRPO variants and math reasoning results | Blog (Emergent Mind) | https://www.emergentmind.com/topics/grpo-based-post-training [4] |
| Vision researcher’s guide to PPO & GRPO | Intuitive explanation of GRPO vs PPO and group-normalized advantages | Blog | https://yugeten.github.io/posts/2025/01/ppogrpo/ [27] |
| GRPO/RLVR GSM8K repo (Qwen2.5) | Public implementation of RLVR+GRPO on GSM8K for small Qwen models, including reward design and configs | GitHub | https://github.com/Mohammadjafari80/GSM8K-RLVR [2] |
| VERL algorithm baselines | GRPO-LoRA performance tables on GSM8K for Qwen2.5 models | Library docs | https://verl.readthedocs.io/en/latest/algo/baseline.html [3] |
| From REINFORCE to Dr. GRPO | Informal notes on policy-gradient estimators, including RLOO and GRPO variants | Blog | https://lancelqf.github.io/note/llm_post_training/ [24] |

These references collectively provide both the theoretical foundation (REINFORCE/RLOO/GRPO analysis) and practical recipes (RLVR GSM8K code, VERL benchmarks, RLHF book) needed to justify and interpret coursework experiments comparing GRPO and RLOO on GSM8K for a Gemma‑scale model.