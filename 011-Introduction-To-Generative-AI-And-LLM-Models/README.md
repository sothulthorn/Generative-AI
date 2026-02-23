# Introduction To Generative AI And LLM Models

## [How ChatGPT Is Trained](https://rpradeepmenon.medium.com/discover-how-chatgpt-is-trained-1f20b9777d1b)

### ChatGPT Training Overview

```
            ┌────────────────────────────────┐
            │  Raw Text Data (Unlabeled)     │
            └────────────────────────────────┘
                          │
                          ▼
            ┌────────────────────────────────┐
            │  Stage 1 – Pretraining         │
            │  (Learn Language Patterns)     │
            └────────────────────────────────┘
                          │
                          ▼
            ┌─────────────────────────────────┐
            │  Stage 2 – Supervised Fine-Tune │
            │  (Human Example Conversations)  │
            └─────────────────────────────────┘
                          │
                          ▼
            ┌───────────────────────────────────────────┐
            │  Stage 3 – Reinforcement Learning (RLHF)  │
            │  (Reward Model + PPO Optimization)        │
            └───────────────────────────────────────────┘
                          │
                          ▼
            ┌────────────────────────────────┐
            │ ChatGPT Model (Aligned & Ready)│
            └────────────────────────────────┘
```

### 🧠 Stage 1 — Generative Pre-Training

**Goal**: Teach the model general language understanding.

- ChatGPT initially begins as a large transformer-based language model.
- It is trained on massive corpora of text data (web pages, books, articles, etc.) in an unsupervised manner.
- The model learns to predict the next word/token in a sentence (next-token prediction).
- This builds a general understanding of grammar, facts, context, and language patterns.
- The result is a powerful base language model, but not yet optimized for conversational alignment.

### 🧑‍🏫 Stage 2 — Supervised Fine-Tuning (SFT)

**Goal**: Specialize the model for conversational and helpful responses.

1. Create human-crafted dialogues
   - Humans act as both user and assistant to generate example conversations.
2. Build a training dataset
   - Conversation histories are aligned with ideal responses as output labels.
3. Fine-tune the pretrained base model
   - Using supervised learning (e.g., stochastic gradient descent), the model parameters are updated to match the human demonstrations.

### 🤖 Stage 3 — Reinforcement Learning from Human Feedback (RLHF)

**Goal**: Align the model with human preferences and make responses more helpful and safe.

1. Reward Model Creation
   - Human evaluators rank multiple model outputs for the same prompt.
   - A reward model learns to score responses that humans prefer.
2. Policy Optimization
   - Using Reinforcement Learning (e.g., Proximal Policy Optimization, PPO), the model’s policy is updated based on reward scores.
   - The objective is to maximize human-aligned rewards rather than raw likelihood.
3. KL Regularization
   - KL-divergence is used to avoid drifting too far from the supervised fine-tuned policy while still learning from the rewards.

### 🏁 Final Model

After completing all three stages:

✔️ The model understands language broadly.

✔️ It can chat and follow instructions.

✔️ It aligns responses with human preferences.
