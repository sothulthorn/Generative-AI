# Transformers - "Attention Is All You Need"

## Table of Contents

1. [Recap: From RNNs to Attention](#1-recap-from-rnns-to-attention)
   - [The Full Journey](#the-full-journey)
   - [What RNN + Attention Still Gets Wrong](#what-rnn--attention-still-gets-wrong)
2. [The Core Idea: Remove Recurrence Entirely](#2-the-core-idea-remove-recurrence-entirely)
3. [Transformer Architecture Overview](#3-transformer-architecture-overview)
   - [Encoder-Decoder Structure](#encoder-decoder-structure)
   - [Architecture Diagram](#architecture-diagram)
4. [Input Representation](#4-input-representation)
   - [Token Embeddings](#token-embeddings)
   - [Positional Encoding](#positional-encoding)
   - [Why Sinusoidal Functions?](#why-sinusoidal-functions)
   - [Final Input](#final-input)
5. [Self-Attention: The Heart of the Transformer](#5-self-attention-the-heart-of-the-transformer)
   - [Intuition: Why Self-Attention?](#intuition-why-self-attention)
   - [Query, Key, Value (Q, K, V)](#query-key-value-q-k-v)
   - [Scaled Dot-Product Attention](#scaled-dot-product-attention)
   - [Step-by-Step Numerical Example](#step-by-step-numerical-example)
   - [Why Scale by $\sqrt{d_k}$?](#why-scale-by-sqrtd_k)
6. [Multi-Head Attention](#6-multi-head-attention)
   - [Why Multiple Heads?](#why-multiple-heads)
   - [Mathematical Formulation](#mathematical-formulation)
   - [Dimension Walkthrough](#dimension-walkthrough)
7. [The Encoder](#7-the-encoder)
   - [Encoder Layer Structure](#encoder-layer-structure)
   - [Add & Norm (Residual Connection + Layer Normalization)](#add--norm-residual-connection--layer-normalization)
   - [Position-Wise Feed-Forward Network](#position-wise-feed-forward-network)
   - [Stacking Encoder Layers](#stacking-encoder-layers)
8. [The Decoder](#8-the-decoder)
   - [Decoder Layer Structure](#decoder-layer-structure)
   - [Masked Multi-Head Self-Attention](#masked-multi-head-self-attention)
   - [Cross-Attention (Encoder-Decoder Attention)](#cross-attention-encoder-decoder-attention)
   - [Stacking Decoder Layers](#stacking-decoder-layers)
9. [Output Generation](#9-output-generation)
   - [Linear Layer + Softmax](#linear-layer--softmax)
10. [Complete Forward Pass Walkthrough](#10-complete-forward-pass-walkthrough)
11. [Training the Transformer](#11-training-the-transformer)
    - [Teacher Forcing](#teacher-forcing)
    - [Loss Function](#loss-function)
    - [Learning Rate Schedule (Warmup)](#learning-rate-schedule-warmup)
    - [Label Smoothing](#label-smoothing)
12. [Why Transformers Work So Well](#12-why-transformers-work-so-well)
13. [Transformer Variants](#13-transformer-variants)
    - [Encoder-Only (BERT)](#encoder-only-bert)
    - [Decoder-Only (GPT)](#decoder-only-gpt)
    - [Encoder-Decoder (T5, BART)](#encoder-decoder-t5-bart)
14. [Summary](#14-summary)
15. [Other References](#15-other-references)

---

## 1. Recap: From RNNs to Attention

### The Full Journey

| #   | Architecture                  | Key Innovation                                  | Remaining Problem               |
| --- | ----------------------------- | ----------------------------------------------- | ------------------------------- |
| 1   | **Simple RNN**                | Sequential memory via hidden state              | Vanishing gradients             |
| 2   | **LSTM / GRU**                | Gates + cell state for long-range memory        | Still sequential                |
| 3   | **Bidirectional RNN**         | Both past and future context                    | Still sequential                |
| 4   | **Seq2Seq (Encoder-Decoder)** | Variable-length input to variable-length output | Fixed context vector bottleneck |
| 5   | **Seq2Seq + Attention**       | Dynamic focus on relevant input positions       | **Still sequential (RNN)**      |
| 6   | **Transformer**               | **Remove RNN entirely — use only attention**    | (Solved)                        |

$$\text{RNN} \xrightarrow{\text{gates}} \text{LSTM} \xrightarrow{\text{bidirectional}} \text{BiRNN} \xrightarrow{\text{seq2seq}} \text{Enc-Dec} \xrightarrow{\text{attention}} \text{Attn Seq2Seq} \xrightarrow{\text{remove RNN}} \textbf{Transformer}$$

### What RNN + Attention Still Gets Wrong

Even with attention, the Seq2Seq architecture has fundamental limitations:

| Problem                   | Description                                                                                                                                                        |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Sequential processing** | RNNs process tokens one at a time: $h_t$ depends on $h_{t-1}$. Token at position 100 must wait for all 99 previous tokens to be processed. **Cannot parallelize.** |
| **Slow training**         | Sequential dependency prevents effective GPU utilization. Modern GPUs excel at parallel computation.                                                               |
| **Limited self-context**  | In the encoder, each token sees its neighbors through recurrence, but there is no mechanism for distant tokens to directly interact.                               |
| **Gradient path length**  | Information between position 1 and position $T$ must travel through $T$ recurrent steps — gradient signal degrades even with LSTM.                                 |

> The Transformer's key insight: **Attention itself is powerful enough to replace recurrence.** If every token can directly attend to every other token, we don't need the RNN chain at all.

---

## 2. The Core Idea: Remove Recurrence Entirely

The Transformer, introduced in **"Attention Is All You Need"** (Vaswani et al., 2017), makes a radical architectural choice:

$$\text{RNN + Attention} \xrightarrow{\text{remove RNN}} \text{Only Attention}$$

| Feature              | RNN + Attention                          | Transformer                              |
| -------------------- | ---------------------------------------- | ---------------------------------------- |
| Sequence processing  | Sequential (token by token)              | **Fully parallel** (all tokens at once)  |
| Self-interaction     | Tokens interact through recurrence chain | Tokens **directly attend** to each other |
| Position information | Implicit in RNN order                    | Explicit **positional encodings**        |
| Max path length      | $O(T)$ steps between distant tokens      | $O(1)$ — direct attention connection     |
| Training speed       | Slow (sequential)                        | **Fast** (parallelizable)                |
| Architecture         | RNN cells + attention layer              | **Only attention + feed-forward layers** |

The Transformer processes **all tokens simultaneously** through self-attention, where every token can directly look at every other token in a single operation. This reduces the maximum path length between any two positions from $O(T)$ (RNN) to $O(1)$.

---

## 3. Transformer Architecture Overview

### Encoder-Decoder Structure

The Transformer retains the high-level Encoder-Decoder structure from Seq2Seq, but replaces all recurrent components with attention and feed-forward layers:

- **Encoder:** Reads the entire input sequence in parallel and produces rich representations
- **Decoder:** Generates the output sequence one token at a time, attending to both its own previous outputs and the encoder's representations

### Architecture Diagram

```
                              OUTPUT
                                ^
                                |
                         [ Linear + Softmax ]
                                ^
                                |
                     +--------------------+
                     |                    |
                     |   DECODER LAYER    |  x N
                     |                    |  (stacked)
                     | +----------------+ |
                     | | Feed-Forward   | |
                     | | Add & Norm     | |
                     | +----------------+ |
                     | | Cross-Attention | |<--------+
                     | | Add & Norm     | |         |
                     | +----------------+ |         |
                     | | Masked Self-   | |         |
                     | | Attention      | |         |
                     | | Add & Norm     | |         |
                     | +----------------+ |         |
                     +--------------------+         |
                              ^                     |
                              |                     |
                     [ Output Embedding             |
                       + Positional Encoding ]      |
                              ^                     |
                              |                     |
                        OUTPUT TOKENS          ENCODER
                        (shifted right)        OUTPUT
                                                    |
                                           +--------------------+
                                           |                    |
                                           |   ENCODER LAYER    |  x N
                                           |                    |  (stacked)
                                           | +----------------+ |
                                           | | Feed-Forward   | |
                                           | | Add & Norm     | |
                                           | +----------------+ |
                                           | | Self-Attention | |
                                           | | Add & Norm     | |
                                           | +----------------+ |
                                           +--------------------+
                                                    ^
                                                    |
                                           [ Input Embedding
                                             + Positional Encoding ]
                                                    ^
                                                    |
                                              INPUT TOKENS
```

**The original paper uses $N = 6$ encoder layers and $N = 6$ decoder layers.**

---

## 4. Input Representation

Since the Transformer has no recurrence, it has no inherent notion of token order. The input representation must explicitly encode **both meaning and position**.

### Token Embeddings

Each input token is converted to a dense vector via a learned embedding matrix, identical to the embedding layer in Seq2Seq:

$$\text{embed}(x_t) \in \mathbb{R}^{d_{\text{model}}}$$

Where $d_{\text{model}} = 512$ in the original paper.

### Positional Encoding

To inject **position information**, a positional encoding vector is added to each token embedding. The original Transformer uses **sinusoidal functions**:

$$\boxed{PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)}$$

$$\boxed{PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)}$$

Where:

- $pos$ is the position of the token in the sequence (0, 1, 2, ...)
- $i$ is the dimension index (0, 1, 2, ..., $d_{\text{model}}/2 - 1$)
- Even dimensions use $\sin$, odd dimensions use $\cos$

**Example** ($d_{\text{model}} = 4$, computing PE for position 0 and position 1):

Position 0:

$$PE_{(0,0)} = \sin\left(\frac{0}{10000^{0/4}}\right) = \sin(0) = 0$$

$$PE_{(0,1)} = \cos\left(\frac{0}{10000^{0/4}}\right) = \cos(0) = 1$$

$$PE_{(0,2)} = \sin\left(\frac{0}{10000^{2/4}}\right) = \sin(0) = 0$$

$$PE_{(0,3)} = \cos\left(\frac{0}{10000^{2/4}}\right) = \cos(0) = 1$$

$$PE_0 = [0, 1, 0, 1]$$

Position 1:

$$PE_{(1,0)} = \sin\left(\frac{1}{10000^{0/4}}\right) = \sin(1) = 0.841$$

$$PE_{(1,1)} = \cos\left(\frac{1}{10000^{0/4}}\right) = \cos(1) = 0.540$$

$$PE_{(1,2)} = \sin\left(\frac{1}{10000^{2/4}}\right) = \sin(0.01) = 0.010$$

$$PE_{(1,3)} = \cos\left(\frac{1}{10000^{2/4}}\right) = \cos(0.01) = 0.999$$

$$PE_1 = [0.841, 0.540, 0.010, 0.999]$$

### Why Sinusoidal Functions?

| Reason                               | Explanation                                                                                                                                                          |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Unique per position**              | Each position gets a distinct encoding pattern                                                                                                                       |
| **Relative positions are learnable** | For any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$. This allows the model to easily learn to attend to relative positions. |
| **Generalizes to unseen lengths**    | Unlike learned positional embeddings, sinusoidal encodings can extrapolate to sequence lengths longer than those seen during training                                |
| **Bounded values**                   | $\sin$ and $\cos$ always output values in $[-1, 1]$, keeping the encoding magnitude controlled                                                                       |

### Final Input

The final input to the encoder (or decoder) is the element-wise **sum** of the token embedding and the positional encoding:

$$\boxed{X_t = \text{Embed}(x_t) + PE_t}$$

| Component           | Carries                                   | Shape                |
| ------------------- | ----------------------------------------- | -------------------- |
| $\text{Embed}(x_t)$ | **Semantic meaning** of the token         | $(d_{\text{model}})$ |
| $PE_t$              | **Position** of the token in the sequence | $(d_{\text{model}})$ |
| $X_t$               | Both meaning and position                 | $(d_{\text{model}})$ |

---

## 5. Self-Attention: The Heart of the Transformer

### Intuition: Why Self-Attention?

Consider the sentence: **"The animal didn't cross the street because it was too tired."**

What does **"it"** refer to? A human instantly knows "it" = "animal" (not "street"), because an animal can be tired but a street cannot.

In a standard RNN, the connection between "it" (position 9) and "animal" (position 2) requires information to flow through 7 recurrent steps. In **self-attention**, "it" can **directly attend** to "animal" in a single step — no matter how far apart they are.

Self-attention lets every token in a sequence **look at every other token** and determine how relevant each one is:

```
  "The"  "animal"  "didn't"  "cross"  "the"  "street"  "because"  "it"  "was"  "too"  "tired"
                                                                     |
                       <============= self-attention ===============>
                       "it" attends to all tokens, focuses most on "animal"
```

### Query, Key, Value (Q, K, V)

Self-attention uses three projections of the input, inspired by information retrieval:

| Concept   | Symbol | Analogy                                      | Role                                         |
| --------- | ------ | -------------------------------------------- | -------------------------------------------- |
| **Query** | $Q$    | A search query: "What am I looking for?"     | The token that is **asking** for information |
| **Key**   | $K$    | An index label: "What do I contain?"         | Each token **advertises** what it has        |
| **Value** | $V$    | The actual content: "Here is my information" | The actual information to be **retrieved**   |

For each token, we compute Q, K, and V by multiplying the input representation by three learned weight matrices:

$$Q = X \cdot W^Q \qquad K = X \cdot W^K \qquad V = X \cdot W^V$$

Where:

- $X \in \mathbb{R}^{T \times d_{\text{model}}}$ — input matrix (all $T$ tokens stacked)
- $W^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$ — query projection weights
- $W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ — key projection weights
- $W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$ — value projection weights

In the original paper: $d_k = d_v = d_{\text{model}} / h = 512 / 8 = 64$, where $h$ is the number of attention heads.

### Scaled Dot-Product Attention

The core attention computation in one formula:

$$\boxed{\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \cdot V}$$

**Breaking this down step by step:**

**Step 1 — Compute dot products between queries and keys:**

$$Q \cdot K^T \in \mathbb{R}^{T \times T}$$

Each entry $(i, j)$ in this matrix is the **dot product** between query $i$ and key $j$, measuring how much token $i$ should attend to token $j$. This produces a $T \times T$ **attention score matrix**.

**Step 2 — Scale by $\sqrt{d_k}$:**

$$\frac{Q \cdot K^T}{\sqrt{d_k}}$$

This prevents the dot products from growing too large (explained below).

**Step 3 — Apply softmax (row-wise):**

$$\text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{T \times T}$$

Each row becomes a probability distribution: row $i$ tells us how much attention token $i$ pays to each other token. All values are positive and each row sums to 1.

**Step 4 — Multiply by values:**

$$\text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \cdot V \in \mathbb{R}^{T \times d_v}$$

Each token's output is a **weighted combination** of all value vectors, weighted by the attention scores. This is the same weighted-sum idea as in Seq2Seq attention, but now **every token attends to every other token** (including itself).

### Step-by-Step Numerical Example

Consider 3 tokens with $d_k = d_v = 4$:

$$Q = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{bmatrix} \quad K = \begin{bmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \\ 1 & 0 & 1 & 0 \end{bmatrix} \quad V = \begin{bmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \end{bmatrix}$$

**Step 1:** $Q \cdot K^T$

$$Q \cdot K^T = \begin{bmatrix} 1 & 1 & 2 \\ 1 & 1 & 0 \\ 1 & 0 & 1 \end{bmatrix}$$

**Step 2:** Scale by $\sqrt{d_k} = \sqrt{4} = 2$

$$\frac{Q \cdot K^T}{\sqrt{d_k}} = \begin{bmatrix} 0.5 & 0.5 & 1.0 \\ 0.5 & 0.5 & 0.0 \\ 0.5 & 0.0 & 0.5 \end{bmatrix}$$

**Step 3:** Softmax (row-wise)

$$\text{softmax} = \begin{bmatrix} 0.27 & 0.27 & 0.46 \\ 0.39 & 0.39 & 0.22 \\ 0.39 & 0.22 & 0.39 \end{bmatrix}$$

Row 1: Token 1 attends most to Token 3 (0.46).
Row 2: Token 2 attends equally to Tokens 1 and 2 (0.39 each).
Row 3: Token 3 attends equally to Tokens 1 and 3 (0.39 each).

**Step 4:** Multiply by $V$

$$\text{Output} = \text{softmax} \cdot V = \begin{bmatrix} 5.38 & 6.38 & 7.38 & 8.38 \\ 3.16 & 4.16 & 5.16 & 6.16 \\ 3.90 & 4.90 & 5.90 & 6.90 \end{bmatrix}$$

> Each output row is a **weighted blend** of the value vectors. Token 1's output is 27% of $V_1$ + 27% of $V_2$ + 46% of $V_3$.

### Why Scale by $\sqrt{d_k}$?

When $d_k$ is large, the dot products $Q \cdot K^T$ can become very large in magnitude. This pushes the softmax into regions where it has **extremely small gradients** (saturation):

$$\text{If } d_k = 512: \quad q \cdot k \approx \mathcal{N}(0, d_k) = \mathcal{N}(0, 512)$$

The variance of the dot product grows linearly with $d_k$. Dividing by $\sqrt{d_k}$ normalizes the variance back to approximately 1:

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{\text{Var}(q \cdot k)}{d_k} = \frac{d_k}{d_k} = 1$$

This keeps the softmax in a regime where gradients are healthy and the model can learn effectively.

---

## 6. Multi-Head Attention

### Why Multiple Heads?

A single attention head can only capture **one type of relationship** at a time. But language has many simultaneous relationships:

- **Syntactic:** "it" refers to "animal" (coreference)
- **Semantic:** "tired" relates to "animal" (property)
- **Positional:** "didn't" modifies "cross" (adjacent dependency)

**Multi-Head Attention** runs $h$ attention operations in parallel, each with its own learned projections, allowing the model to attend to information from **different representation subspaces** simultaneously:

$$\text{head}_i = \text{Attention}(Q W_i^Q, \; K W_i^K, \; V W_i^V)$$

### Mathematical Formulation

$$\boxed{\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W^O}$$

Where each head is:

$$\boxed{\text{head}_i = \text{Attention}(X W_i^Q, \; X W_i^K, \; X W_i^V)}$$

| Parameter | Shape                                   | Description                            |
| --------- | --------------------------------------- | -------------------------------------- |
| $W_i^Q$   | $(d_{\text{model}} \times d_k)$         | Query projection for head $i$          |
| $W_i^K$   | $(d_{\text{model}} \times d_k)$         | Key projection for head $i$            |
| $W_i^V$   | $(d_{\text{model}} \times d_v)$         | Value projection for head $i$          |
| $W^O$     | $(h \cdot d_v \times d_{\text{model}})$ | Output projection (combines all heads) |

### Dimension Walkthrough

Original paper: $d_{\text{model}} = 512$, $h = 8$ heads, $d_k = d_v = 512 / 8 = 64$

```
Input X: (T x 512)
         |
    Split into 8 heads
         |
   +-----------+-----------+-----+-----------+
   |           |           |     |           |
 Head 1     Head 2     Head 3  ...  Head 8
Q:(Tx64)   Q:(Tx64)   Q:(Tx64)    Q:(Tx64)
K:(Tx64)   K:(Tx64)   K:(Tx64)    K:(Tx64)
V:(Tx64)   V:(Tx64)   V:(Tx64)    V:(Tx64)
   |           |           |           |
Attn:(Tx64) Attn:(Tx64) Attn:(Tx64) Attn:(Tx64)
   |           |           |           |
   +-----------+-----------+-----------+
                     |
               Concat: (T x 512)
                     |
                W^O: (512 x 512)
                     |
               Output: (T x 512)
```

> Despite having $h$ heads, the total computation is roughly the same as a single-head attention with full dimensionality, because each head operates on a $d_k = d_{\text{model}} / h$ dimensional space. The parallel heads are computationally efficient.

**Total Multi-Head Attention parameters:**

$$h \times (d_{\text{model}} \times d_k + d_{\text{model}} \times d_k + d_{\text{model}} \times d_v) + h \cdot d_v \times d_{\text{model}}$$

$$= h \times 3 \times d_{\text{model}} \times d_k + d_{\text{model}}^2 = 3 \times d_{\text{model}}^2 + d_{\text{model}}^2 = 4 \times d_{\text{model}}^2$$

With $d_{\text{model}} = 512$: approximately **1,048,576** parameters per Multi-Head Attention layer.

---

## 7. The Encoder

### Encoder Layer Structure

Each encoder layer has **two sub-layers**:

```
          Output (T x d_model)
               ^
               |
      +------------------+
      |  Add & Norm      |  <--- Residual + LayerNorm
      +------------------+
               ^
               |
      +------------------+
      |  Feed-Forward    |  <--- Two linear layers + ReLU
      |  Network (FFN)   |
      +------------------+
               ^
               |
      +------------------+
      |  Add & Norm      |  <--- Residual + LayerNorm
      +------------------+
               ^
               |
      +------------------+
      |  Multi-Head      |
      |  Self-Attention  |  <--- Q=K=V=X (attend to all input tokens)
      +------------------+
               ^
               |
          Input (T x d_model)
```

### Add & Norm (Residual Connection + Layer Normalization)

Every sub-layer is wrapped with a **residual connection** followed by **layer normalization**:

$$\boxed{\text{LayerNorm}(X + \text{SubLayer}(X))}$$

**Residual Connection** ($X + \text{SubLayer}(X)$):

The input $X$ is added directly to the sub-layer's output. This creates a **shortcut path** that:

- Allows gradients to flow directly through the addition (avoids vanishing gradients)
- Lets the sub-layer learn **refinements** rather than complete transformations
- Makes deep networks (6+ layers) trainable

**Layer Normalization** ($\text{LayerNorm}$):

Normalizes across the feature dimension (not the batch dimension):

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where $\mu$ and $\sigma^2$ are the mean and variance computed across the $d_{\text{model}}$ features for each token, and $\gamma, \beta$ are learnable scale and shift parameters.

> Residual connections require the sub-layer output to have the same dimension as the input. This is why all sub-layers and embeddings in the Transformer produce outputs of dimension $d_{\text{model}} = 512$.

### Position-Wise Feed-Forward Network

The second sub-layer is a simple **two-layer fully connected network** applied to each position independently:

$$\boxed{\text{FFN}(x) = \text{ReLU}(x \cdot W_1 + b_1) \cdot W_2 + b_2}$$

| Parameter | Shape                              | Description                              |
| --------- | ---------------------------------- | ---------------------------------------- |
| $W_1$     | $(d_{\text{model}} \times d_{ff})$ | First linear projection (expand)         |
| $b_1$     | $(d_{ff})$                         | First bias                               |
| $W_2$     | $(d_{ff} \times d_{\text{model}})$ | Second linear projection (compress back) |
| $b_2$     | $(d_{\text{model}})$               | Second bias                              |

In the original paper: $d_{ff} = 2048$ (4x expansion of $d_{\text{model}} = 512$).

The FFN creates a **bottleneck**: expand to a higher dimension (2048), apply non-linearity (ReLU), then compress back to the original dimension (512). This gives each position a chance to do **non-linear feature transformation** beyond what attention alone provides.

$$x \in \mathbb{R}^{512} \xrightarrow{W_1} \mathbb{R}^{2048} \xrightarrow{\text{ReLU}} \mathbb{R}^{2048} \xrightarrow{W_2} \mathbb{R}^{512}$$

> "Position-wise" means the same FFN is applied to each token independently (like a 1x1 convolution). There is no interaction between tokens in this sub-layer — that's what self-attention is for.

### Stacking Encoder Layers

The encoder stacks $N = 6$ identical layers. Each layer refines the representations:

$$\text{Layer 1:} \quad X^{(0)} \rightarrow X^{(1)}$$

$$\text{Layer 2:} \quad X^{(1)} \rightarrow X^{(2)}$$

$$\vdots$$

$$\text{Layer 6:} \quad X^{(5)} \rightarrow X^{(6)}$$

Lower layers tend to capture **local/syntactic features**, while higher layers capture **global/semantic features**. The final output $X^{(6)} \in \mathbb{R}^{T \times d_{\text{model}}}$ is the encoder's rich representation of the input, passed to the decoder.

---

## 8. The Decoder

### Decoder Layer Structure

Each decoder layer has **three sub-layers**:

```
          Output (n x d_model)
               ^
               |
      +------------------+
      |  Add & Norm      |
      +------------------+
               ^
               |
      +------------------+
      |  Feed-Forward    |
      |  Network (FFN)   |
      +------------------+
               ^
               |
      +------------------+
      |  Add & Norm      |
      +------------------+
               ^
               |
      +------------------+
      |  Cross-Attention |  <--- Q from decoder, K & V from encoder
      |  (Enc-Dec Attn)  |
      +------------------+
               ^
               |
      +------------------+
      |  Add & Norm      |
      +------------------+
               ^
               |
      +------------------+
      |  Masked Multi-   |
      |  Head Self-Attn  |  <--- Q=K=V from decoder (masked)
      +------------------+
               ^
               |
          Input (n x d_model)
```

### Masked Multi-Head Self-Attention

The decoder's self-attention is **masked** to prevent positions from attending to **future tokens**. This is essential because during inference, future tokens don't exist yet — the model generates them one at a time.

**The mask:** An upper-triangular matrix of $-\infty$ values added to the attention scores before softmax:

$$\text{Mask} = \begin{bmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{bmatrix}$$

$$\text{MaskedAttention} = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + \text{Mask}\right) V$$

After adding $-\infty$ and applying softmax, the masked positions become **0** (since $e^{-\infty} = 0$):

$$\text{softmax with mask} = \begin{bmatrix} 1.0 & 0 & 0 & 0 \\ 0.6 & 0.4 & 0 & 0 \\ 0.2 & 0.3 & 0.5 & 0 \\ 0.1 & 0.2 & 0.3 & 0.4 \end{bmatrix}$$

| Position | Can Attend To     | Cannot See     |
| -------- | ----------------- | -------------- |
| Token 1  | Token 1           | Tokens 2, 3, 4 |
| Token 2  | Tokens 1, 2       | Tokens 3, 4    |
| Token 3  | Tokens 1, 2, 3    | Token 4        |
| Token 4  | Tokens 1, 2, 3, 4 | Nothing        |

> This ensures the decoder is **autoregressive**: the prediction for position $i$ depends only on the known outputs at positions less than $i$. This maintains the causal property needed for generation.

### Cross-Attention (Encoder-Decoder Attention)

The second sub-layer is **cross-attention**, which is the same mechanism as the attention in Seq2Seq, but implemented as multi-head attention:

$$\text{CrossAttention:} \quad Q = \text{Decoder}, \quad K = V = \text{Encoder output}$$

| Source               | Role                                   | Meaning                                                                    |
| -------------------- | -------------------------------------- | -------------------------------------------------------------------------- |
| $Q$ from **Decoder** | "What am I looking for?"               | The decoder asks what input information is relevant for the current output |
| $K$ from **Encoder** | "What does each input contain?"        | The encoder advertises what each input position holds                      |
| $V$ from **Encoder** | "Here is the actual input information" | The encoder provides its rich representations                              |

This is where the decoder **looks at the input sentence**. It is functionally equivalent to the attention mechanism in Seq2Seq, but with multi-head attention for richer interactions.

> Cross-attention is identical in mechanism to self-attention — the only difference is that $K$ and $V$ come from a **different sequence** (encoder) than $Q$ (decoder).

### Stacking Decoder Layers

Like the encoder, $N = 6$ decoder layers are stacked. Each layer receives:

1. The output of the previous decoder layer
2. The encoder's final output (same for all decoder layers — $K$ and $V$ from the encoder are the same at every layer)

---

## 9. Output Generation

### Linear Layer + Softmax

After the final decoder layer, the output is passed through a **linear projection** followed by **softmax** to produce probabilities over the target vocabulary:

$$\boxed{\hat{y}_t = \text{softmax}(W_{\text{vocab}} \cdot d_t + b_{\text{vocab}})}$$

Where:

- $d_t \in \mathbb{R}^{d_{\text{model}}}$ is the decoder output at position $t$
- $W_{\text{vocab}} \in \mathbb{R}^{d_{\text{model}} \times |V|}$ projects to vocabulary size
- $|V|$ is the target vocabulary size (e.g., 37,000 in the original paper)

The predicted token is:

$$w_t = \arg\max(\hat{y}_t)$$

> In the original paper, the output embedding matrix and $W_{\text{vocab}}$ share the same weights (weight tying), reducing parameters.

---

## 10. Complete Forward Pass Walkthrough

Let's trace a complete forward pass for translating **"I love you"** to **"Je t'aime"**.

### Phase 1 — Encoder

**Input tokens:** ["I", "love", "you"]

**Step 1: Embedding + Positional Encoding**

$$X_1 = \text{Embed}(\text{"I"}) + PE_0$$

$$X_2 = \text{Embed}(\text{"love"}) + PE_1$$

$$X_3 = \text{Embed}(\text{"you"}) + PE_2$$

$$X^{(0)} = \begin{bmatrix} X_1 \\ X_2 \\ X_3 \end{bmatrix} \in \mathbb{R}^{3 \times 512}$$

**Step 2: Encoder Layer 1 — Self-Attention**

All three tokens attend to each other simultaneously:

$$Q = X^{(0)} W^Q, \quad K = X^{(0)} W^K, \quad V = X^{(0)} W^V$$

$$\text{SelfAttn} = \text{softmax}\left(\frac{QK^T}{\sqrt{64}}\right) V$$

"love" can directly see both "I" and "you", understanding the full relationship.

$$X^{(0.5)} = \text{LayerNorm}(X^{(0)} + \text{MultiHeadSelfAttn}(X^{(0)}))$$

**Step 3: Encoder Layer 1 — Feed-Forward**

$$X^{(1)} = \text{LayerNorm}(X^{(0.5)} + \text{FFN}(X^{(0.5)}))$$

**Steps 4-13: Encoder Layers 2-6** (same structure, progressively refining representations)

**Final encoder output:** $\text{Enc} = X^{(6)} \in \mathbb{R}^{3 \times 512}$

### Phase 2 — Decoder (generating "Je")

**Decoder input:** [`<SOS>`] (shifted right target)

**Step 1: Embedding + Positional Encoding**

$$D^{(0)} = [\text{Embed}(\text{SOS}) + PE_0] \in \mathbb{R}^{1 \times 512}$$

**Step 2: Masked Self-Attention**

Only one token, so no masking effect. `<SOS>` attends to itself.

**Step 3: Cross-Attention**

$$Q = D^{(0.5)} \quad (\text{from decoder})$$

$$K = \text{Enc}, \quad V = \text{Enc} \quad (\text{from encoder})$$

The decoder "looks at" the encoder's representation of "I love you" to decide what the first output word should be.

**Step 4: Feed-Forward + Final Output**

$$\hat{y}_1 = \text{softmax}(W_{\text{vocab}} \cdot d_1) \rightarrow \textbf{"Je"}$$

### Phase 3 — Decoder (generating "t'aime")

**Decoder input:** [`<SOS>`, "Je"]

**Step 2: Masked Self-Attention**

"Je" can attend to `<SOS>` and itself. `<SOS>` can only attend to itself.

**Step 3: Cross-Attention**

Both decoder positions attend to the full encoder output, with attention weights likely focusing on "love" and "you".

**Step 4: Output**

$$\hat{y}_2 = \text{softmax}(W_{\text{vocab}} \cdot d_2) \rightarrow \textbf{"t'aime"}$$

### Phase 4 — Decoder (generating `<EOS>`)

**Decoder input:** [`<SOS>`, "Je", "t'aime"]

$\hat{y}_3 \rightarrow$ **\<EOS\>** (STOP)

**Final output:** "Je t'aime"

---

## 11. Training the Transformer

### Teacher Forcing

Like Seq2Seq, the Transformer uses **teacher forcing** during training. The decoder receives the **ground truth shifted right** as input:

| Decoder Input (shifted right) | Target Output     |
| ----------------------------- | ----------------- |
| `<SOS>` Je t'aime             | Je t'aime `<EOS>` |

All positions are computed **in parallel** during training (unlike RNN-based models). The mask ensures that position $i$ can only see positions $< i$.

> This is a major advantage: during training, the entire target sequence is processed in **one forward pass**, compared to $n$ sequential steps in RNN-based Seq2Seq. This massively speeds up training.

### Loss Function

Cross-entropy loss summed over all output positions:

$$\boxed{\mathcal{L} = -\sum_{t=1}^{n} \log P(y_t^{\text{true}} | y_{<t}, \mathbf{x})}$$

### Learning Rate Schedule (Warmup)

The original paper uses a custom learning rate schedule that **warms up linearly** then **decays**:

$$\boxed{lr = d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5}, \; \text{step} \cdot \text{warmup\_steps}^{-1.5})}$$

```
  Learning
    Rate
     ^
     |        *  *
     |      *      *
     |    *          *
     |   *             *
     |  *                *
     | *                   *  *  *  *
     |*
     +---------------------------------------->
     0   warmup_steps            Training Steps
          (4000)
```

| Phase                        | Behavior                                        | Reason                                                                       |
| ---------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------- |
| **Warmup** (0 to 4000 steps) | Learning rate increases linearly                | Prevents large unstable updates early in training when parameters are random |
| **Decay** (4000+ steps)      | Learning rate decreases as $\text{step}^{-0.5}$ | Gradually reduces updates for fine-grained convergence                       |

### Label Smoothing

The original paper uses **label smoothing** with $\epsilon = 0.1$:

Instead of a hard one-hot target $[0, 0, 1, 0, 0]$, the target becomes:

$$[0.02, 0.02, 0.92, 0.02, 0.02]$$

A small probability $\epsilon / |V|$ is distributed across all classes, and the true class gets $1 - \epsilon + \epsilon / |V|$.

> Label smoothing prevents the model from becoming overconfident and improves generalization, at the cost of slightly higher perplexity.

---

## 12. Why Transformers Work So Well

| Advantage                     | Explanation                                                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Full parallelism**          | All tokens processed simultaneously during training. GPUs can operate at full capacity.                                        |
| **$O(1)$ path length**        | Any two tokens are connected through a single attention step, regardless of distance. No vanishing gradients over long ranges. |
| **Multi-head diversity**      | Different heads learn different relationship types (syntax, semantics, coreference, etc.) simultaneously.                      |
| **Deep refinement**           | 6+ stacked layers progressively build richer representations, from surface features to deep semantics.                         |
| **Residual connections**      | Gradients flow freely through skip connections, enabling training of very deep models.                                         |
| **Scalability**               | Architecture scales well with more data, more parameters, and more compute — leading directly to large language models.        |
| **No information bottleneck** | Cross-attention gives the decoder direct access to every encoder position at every layer, not just a single context vector.    |

### Complexity Comparison

| Model                        | Sequential Ops per Layer | Max Path Length | Computation per Layer |
| ---------------------------- | ------------------------ | --------------- | --------------------- |
| RNN                          | $O(T)$                   | $O(T)$          | $O(T \cdot d^2)$      |
| Transformer (Self-Attention) | $O(1)$                   | $O(1)$          | $O(T^2 \cdot d)$      |

> The Transformer trades **sequential operations** (which RNNs have) for **quadratic attention computation** ($T^2$). For typical NLP sequence lengths ($T < 1000$), this is a favorable trade because $O(1)$ parallelism more than compensates for the $T^2$ attention cost.

---

## 13. Transformer Variants

The original Transformer is an encoder-decoder model. Subsequent work discovered that using **only the encoder** or **only the decoder** is highly effective for different tasks.

### Encoder-Only (BERT)

**BERT** (Bidirectional Encoder Representations from Transformers, 2018):

- Uses only the **encoder** stack
- **Bidirectional self-attention** (no mask — every token sees every other token)
- Pre-trained with **Masked Language Modeling** (predict masked tokens) and **Next Sentence Prediction**
- Fine-tuned for downstream tasks: classification, NER, question answering

```
         [CLS]  Token1  Token2  [SEP]  Token3  Token4  [SEP]
           |       |       |      |       |       |      |
         [Encoder Layer x 12/24]  (full bidirectional attention)
           |       |       |      |       |       |      |
          H_0     H_1     H_2    H_3    H_4     H_5    H_6
           |
       Classification
```

**Best for:** Understanding tasks (classification, NER, semantic similarity, extractive QA).

### Decoder-Only (GPT)

**GPT** (Generative Pre-trained Transformer, 2018-present):

- Uses only the **decoder** stack (with masked/causal self-attention)
- **Unidirectional** — each token can only attend to previous tokens
- Pre-trained with **next-token prediction** (language modeling)
- Generates text autoregressively

```
  Token1  Token2  Token3  Token4
    |       |       |       |
  [Decoder Layer x N]  (causal masked attention)
    |       |       |       |
    v       v       v       v
  Token2  Token3  Token4  Token5
  (next)  (next)  (next)  (next)
```

**Best for:** Generation tasks (text completion, dialogue, code generation, reasoning). GPT-3, GPT-4, Claude, and most modern LLMs use this architecture.

### Encoder-Decoder (T5, BART)

**T5** (Text-to-Text Transfer Transformer, 2019) and **BART** (2019):

- Use the **full encoder-decoder** architecture (closest to the original Transformer)
- Frame all NLP tasks as text-to-text: input text $\rightarrow$ output text
- Pre-trained with various denoising objectives

**Best for:** Sequence-to-sequence tasks (translation, summarization, generative QA).

### Comparison

| Variant     | Architecture      | Attention Type                     | Pre-training          | Best For                             |
| ----------- | ----------------- | ---------------------------------- | --------------------- | ------------------------------------ |
| **BERT**    | Encoder only      | Bidirectional (full)               | Masked LM             | Understanding (NER, classification)  |
| **GPT**     | Decoder only      | Causal (masked)                    | Next-token prediction | Generation (text, code, dialogue)    |
| **T5/BART** | Encoder + Decoder | Bidirectional (enc) + Causal (dec) | Denoising             | Seq2Seq (translation, summarization) |

---

## 14. Summary

### The Core Innovation

$$\text{RNN + Attention} \xrightarrow{\text{remove recurrence}} \text{Transformer (Attention Is All You Need)}$$

### Architecture at a Glance

$$
\boxed{\begin{aligned}
&\text{Input:} & X &= \text{Embed}(x) + PE && \text{Token meaning + position} \\[8pt]
&\text{Self-Attention:} & \text{Attn}(Q,K,V) &= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V && \text{Every token sees every token} \\[8pt]
&\text{Multi-Head:} & \text{MH}(Q,K,V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O && \text{Multiple relationship types} \\[8pt]
&\text{Encoder:} & & \text{Self-Attn} \rightarrow \text{FFN} \quad (\times N) && \text{Rich input representations} \\[8pt]
&\text{Decoder:} & & \text{Masked Self-Attn} \rightarrow \text{Cross-Attn} \rightarrow \text{FFN} \quad (\times N) && \text{Autoregressive generation} \\[8pt]
&\text{Output:} & \hat{y}_t &= \text{softmax}(W_{\text{vocab}} \cdot d_t) && \text{Probability over vocabulary}
\end{aligned}}
$$

### Key Numbers (Original Paper)

| Hyperparameter                 | Value       |
| ------------------------------ | ----------- |
| $d_{\text{model}}$             | 512         |
| $h$ (attention heads)          | 8           |
| $d_k = d_v$                    | 64          |
| $d_{ff}$ (FFN inner dimension) | 2048        |
| $N$ (encoder/decoder layers)   | 6           |
| Warmup steps                   | 4000        |
| Label smoothing $\epsilon$     | 0.1         |
| Total parameters               | ~65 million |

### The Full Evolution

$$\underbrace{\text{RNN}}_{\text{sequential}} \rightarrow \underbrace{\text{LSTM}}_{\text{long memory}} \rightarrow \underbrace{\text{Seq2Seq}}_{\text{variable length}} \rightarrow \underbrace{\text{Attention}}_{\text{dynamic focus}} \rightarrow \underbrace{\textbf{Transformer}}_{\text{parallel + attention only}} \rightarrow \underbrace{\text{BERT, GPT, LLMs}}_{\text{scale up}}$$

---

## 15. Other References

[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
