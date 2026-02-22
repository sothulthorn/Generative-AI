# Attention Mechanism in Seq2Seq Networks

## Table of Contents

1. [Recap: The Encoder-Decoder Bottleneck](#1-recap-the-encoder-decoder-bottleneck)
   - [What Seq2Seq Does Well](#what-seq2seq-does-well)
   - [Where It Fails](#where-it-fails)
   - [BLEU Score Decay](#bleu-score-decay)
2. [The Core Idea of Attention](#2-the-core-idea-of-attention)
   - [Human Analogy](#human-analogy)
   - [Seq2Seq Without vs With Attention](#seq2seq-without-vs-with-attention)
3. [Encoder with Bidirectional RNN](#3-encoder-with-bidirectional-rnn)
   - [Why Bidirectional?](#why-bidirectional)
   - [Annotations (Encoder Hidden States)](#annotations-encoder-hidden-states)
4. [The Attention Mechanism Step-by-Step](#4-the-attention-mechanism-step-by-step)
   - [Step 1: Compute Alignment Scores](#step-1-compute-alignment-scores)
   - [Step 2: Compute Attention Weights (Softmax)](#step-2-compute-attention-weights-softmax)
   - [Step 3: Compute Context Vector](#step-3-compute-context-vector)
   - [Step 4: Decoder Generates Output](#step-4-decoder-generates-output)
5. [Complete Mathematical Formulation](#5-complete-mathematical-formulation)
   - [Encoder Equations](#encoder-equations)
   - [Alignment Score Function](#alignment-score-function)
   - [Attention Weights](#attention-weights)
   - [Context Vector](#context-vector)
   - [Decoder Equations](#decoder-equations)
   - [Output Prediction](#output-prediction)
6. [Types of Attention](#6-types-of-attention)
   - [Bahdanau Attention (Additive)](#bahdanau-attention-additive)
   - [Luong Attention (Multiplicative)](#luong-attention-multiplicative)
   - [Comparison Table](#comparison-table)
7. [Full Walkthrough: "Hello What's Up" to "Gracias"](#7-full-walkthrough-hello-whats-up-to-gracias)
8. [Visualizing Attention: The Alignment Matrix](#8-visualizing-attention-the-alignment-matrix)
9. [Training with Attention](#9-training-with-attention)
   - [Loss Function](#loss-function)
   - [Gradient Flow](#gradient-flow)
10. [Why Attention Solves the Bottleneck](#10-why-attention-solves-the-bottleneck)
11. [Limitations of Attention in RNNs](#11-limitations-of-attention-in-rnns)
12. [From Attention to Transformers (Preview)](#12-from-attention-to-transformers-preview)
13. [Summary](#13-summary)

---

## 1. Recap: The Encoder-Decoder Bottleneck

### What Seq2Seq Does Well

The standard Encoder-Decoder architecture (Seq2Seq) maps an input sequence to an output sequence of arbitrary length:

$$\text{Encoder:} \quad (x_1, x_2, \ldots, x_T) \xrightarrow{\text{LSTM}} (h_T^{\text{enc}}, C_T^{\text{enc}}) = \text{Context Vector}$$

$$\text{Decoder:} \quad \text{Context Vector} \xrightarrow{\text{LSTM}} (y_1, y_2, \ldots, y_n)$$

This works well for **short sequences** (5-20 tokens) where the fixed-size context vector can faithfully represent the input.

### Where It Fails

The entire input sentence is **compressed into a single fixed-size vector**:

$$\underbrace{(x_1, x_2, \ldots, x_T)}_{\text{Could be 100 words}} \xrightarrow{\text{squeeze}} \underbrace{(h_T, C_T)}_{\text{e.g., 256 dims}}$$

Problems:
- Information from **early tokens gets diluted** as the sequence grows
- The decoder receives the **same static context** for every output token, regardless of which part of the input is most relevant
- For longer sentences, critical details are **lost** in the compression

### BLEU Score Decay

Research (Cho et al., 2014) demonstrated that translation quality degrades sharply with sentence length:

```
  BLEU
  Score
    ^
    |     *  *
    |   *      *
    |  *        *
    | *          *
    |*             *
    |                *
    |                  *  *  *
    +----------------------------->
      10  20  30  40  50  60
           Sentence Length
```

> The root cause: the context vector is a **fixed-capacity bottleneck**. No matter how long the input is, all meaning must pass through the same narrow channel.

---

## 2. The Core Idea of Attention

**Attention** allows the decoder to **look back at all encoder hidden states** at every decoding step, rather than relying on a single compressed context vector. At each step, the decoder learns to **focus on** (attend to) the most relevant parts of the input.

### Human Analogy

When a human translator translates a sentence, they don't:
1. Read the entire source sentence
2. Compress it into a single thought
3. Translate from that single thought

Instead, they:
1. Read the entire source sentence
2. For each word they write, **look back at specific parts** of the source
3. Focus on the **most relevant words** for the current translation

This is exactly what the Attention Mechanism does.

### Seq2Seq Without vs With Attention

**Without Attention** (basic Seq2Seq):

```
Encoder:  [h1]-->[h2]-->[h3]-->[h4]
                                  |
                            Context Vector (h4 only)
                                  |
                                  v
Decoder:                    [s1]-->[s2]-->[s3]
                             |      |      |
                             v      v      v
                            y_1    y_2    y_3
```

The decoder sees **only the final encoder state** $h_4$. All information must pass through this single vector.

**With Attention:**

```
Encoder:  [h1]-->[h2]-->[h3]-->[h4]
            |      |      |      |
            +------+------+------+---- ALL states available
            |      |      |      |
          [Attention weights change at each decoder step]
            |      |      |      |
            v      v      v      v
Decoder:  [s1]-->[s2]-->[s3]
           |      |      |
           v      v      v
          y_1    y_2    y_3
```

At each decoding step, the decoder can access **every encoder hidden state** $(h_1, h_2, \ldots, h_T)$, with learned weights determining which states to focus on.

| Feature | Without Attention | With Attention |
|---|---|---|
| Decoder input | Single fixed context vector | **Different** context vector at each step |
| Encoder states used | Only $h_T$ (last) | **All** $h_1, h_2, \ldots, h_T$ |
| Focus | Same for every output token | **Shifts** to relevant input for each output |
| Long sentences | Information lost | Information **preserved** via direct access |

---

## 3. Encoder with Bidirectional RNN

### Why Bidirectional?

The Attention Mechanism was originally proposed (Bahdanau et al., 2014) with a **Bidirectional RNN** encoder. This ensures that each encoder hidden state captures context from **both directions**:

- $\overrightarrow{h_t}$ captures $(x_1, \ldots, x_t)$ — past context
- $\overleftarrow{h_t}$ captures $(x_t, \ldots, x_T)$ — future context

```
  Forward:  -->[h1]-->[h2]-->[h3]-->
                |       |       |
  Input:       x_1     x_2     x_3
                |       |       |
  Backward: <--[h1]<--[h2]<--[h3]<--
```

### Annotations (Encoder Hidden States)

Each encoder position produces an **annotation** $h_j$ by concatenating the forward and backward hidden states:

$$h_j = [\overrightarrow{h_j} \; ; \; \overleftarrow{h_j}]$$

| Annotation | Forward Component | Backward Component | Full Context |
|---|---|---|---|
| $h_1$ | $\overrightarrow{h_1}$: context from $x_1$ | $\overleftarrow{h_1}$: context from $x_1, x_2, \ldots, x_T$ | Entire sequence, centered around $x_1$ |
| $h_2$ | $\overrightarrow{h_2}$: context from $x_1, x_2$ | $\overleftarrow{h_2}$: context from $x_2, \ldots, x_T$ | Entire sequence, centered around $x_2$ |
| $h_T$ | $\overrightarrow{h_T}$: context from $x_1, \ldots, x_T$ | $\overleftarrow{h_T}$: context from $x_T$ | Entire sequence, centered around $x_T$ |

> Each annotation $h_j$ has a **strong focus on the parts surrounding the $j$-th word** of the input, while still carrying information about the whole sentence. This is what makes attention so powerful — the decoder can access position-specific representations.

---

## 4. The Attention Mechanism Step-by-Step

At each decoder time step $i$, the attention mechanism computes a **fresh context vector** $c_i$ through four steps.

### Step 1: Compute Alignment Scores

The **alignment score** $e_{i,j}$ measures how well the input at position $j$ matches the output at position $i$. It is computed by a small **feed-forward neural network** (called the alignment model):

$$\boxed{e_{i,j} = a(s_{i-1}, h_j)}$$

Where:
- $s_{i-1}$ is the **previous decoder hidden state** (what the decoder has generated so far)
- $h_j$ is the **encoder annotation** at position $j$ (representation of input word $j$)
- $a$ is a learned function (a small neural network)

This score answers the question: **"How relevant is input position $j$ for generating the current output at position $i$?"**

**Example:** For input "Hello What's Up" (3 positions) at decoder step $i = 1$:

| Score | Computation | Meaning |
|---|---|---|
| $e_{1,1}$ | $a(s_0, h_1)$ | How relevant is "Hello" for generating $y_1$? |
| $e_{1,2}$ | $a(s_0, h_2)$ | How relevant is "What's" for generating $y_1$? |
| $e_{1,3}$ | $a(s_0, h_3)$ | How relevant is "Up" for generating $y_1$? |

The alignment model $a$ is typically a single-layer feed-forward network:

$$e_{i,j} = v_a^T \cdot \tanh(W_a \cdot s_{i-1} + U_a \cdot h_j)$$

Where $W_a$, $U_a$, and $v_a$ are **learnable parameters** trained jointly with the rest of the model.

### Step 2: Compute Attention Weights (Softmax)

The raw alignment scores are converted into a **probability distribution** using softmax:

$$\boxed{\alpha_{i,j} = \frac{\exp(e_{i,j})}{\sum_{k=1}^{T} \exp(e_{i,k})}}$$

This ensures:
- All weights are **positive**: $\alpha_{i,j} > 0$
- All weights **sum to 1**: $\sum_{j=1}^{T} \alpha_{i,j} = 1$
- The weights form a valid **probability distribution** over input positions

**Example:** Continuing from above:

$$[e_{1,1}, e_{1,2}, e_{1,3}] = [2.1, 0.5, 0.3]$$

$$\alpha_{1,1} = \frac{e^{2.1}}{e^{2.1} + e^{0.5} + e^{0.3}} = \frac{8.17}{8.17 + 1.65 + 1.35} = \frac{8.17}{11.17} = \mathbf{0.73}$$

$$\alpha_{1,2} = \frac{e^{0.5}}{11.17} = \frac{1.65}{11.17} = \mathbf{0.15}$$

$$\alpha_{1,3} = \frac{e^{0.3}}{11.17} = \frac{1.35}{11.17} = \mathbf{0.12}$$

$$[\alpha_{1,1}, \alpha_{1,2}, \alpha_{1,3}] = [0.73, 0.15, 0.12]$$

> The decoder is paying **73% attention to "Hello"**, 15% to "What's", and 12% to "Up" when generating the first output word. These weights **change** at every decoder step.

### Step 3: Compute Context Vector

The context vector $c_i$ is a **weighted sum** of all encoder annotations, using the attention weights:

$$\boxed{c_i = \sum_{j=1}^{T} \alpha_{i,j} \cdot h_j}$$

**Example:**

$$c_1 = 0.73 \cdot h_1 + 0.15 \cdot h_2 + 0.12 \cdot h_3$$

This is a **soft selection** of encoder states — instead of picking one state (hard attention), the context vector is a smooth blend weighted by relevance.

> **Critical difference from basic Seq2Seq:** In basic Seq2Seq, there is ONE context vector for the entire decoding. With attention, there is a **DIFFERENT context vector $c_i$ for each decoder step $i$**. Each one is custom-tailored to focus on the most relevant input positions for that specific output token.

### Step 4: Decoder Generates Output

The decoder uses the context vector $c_i$ together with its previous state $s_{i-1}$ and previous output $y_{i-1}$ to:

1. **Update its hidden state:**

$$s_i = f(s_{i-1}, y_{i-1}, c_i)$$

2. **Predict the output token:**

$$\hat{y}_i = \text{softmax}(W_o \cdot [s_i \; ; \; c_i] + b_o)$$

The context vector is used in **both** the state update and the output prediction, giving the decoder maximum access to the relevant input information.

---

## 5. Complete Mathematical Formulation

### Encoder Equations

Using a Bidirectional RNN (LSTM or GRU):

**Forward pass** ($t = 1$ to $T$):

$$\overrightarrow{h_t} = \text{LSTM}_{\overrightarrow{\text{enc}}}(x_t, \overrightarrow{h_{t-1}})$$

**Backward pass** ($t = T$ to $1$):

$$\overleftarrow{h_t} = \text{LSTM}_{\overleftarrow{\text{enc}}}(x_t, \overleftarrow{h_{t+1}})$$

**Annotation (concatenation):**

$$\boxed{h_j = [\overrightarrow{h_j} \; ; \; \overleftarrow{h_j}] \quad \in \mathbb{R}^{2n_h}}$$

Where $n_h$ is the hidden size of each direction.

### Alignment Score Function

The alignment model $a$ computes a scalar score for each encoder-decoder position pair:

$$\boxed{e_{i,j} = v_a^T \cdot \tanh(W_a \cdot s_{i-1} + U_a \cdot h_j + b_a)}$$

| Parameter | Shape | Role |
|---|---|---|
| $s_{i-1}$ | $(n_s \times 1)$ | Previous decoder hidden state |
| $h_j$ | $(2n_h \times 1)$ | Encoder annotation at position $j$ |
| $W_a$ | $(d_a \times n_s)$ | Projects decoder state into alignment space |
| $U_a$ | $(d_a \times 2n_h)$ | Projects encoder annotation into alignment space |
| $b_a$ | $(d_a \times 1)$ | Bias |
| $v_a$ | $(d_a \times 1)$ | Collapses the $d_a$-dimensional vector into a scalar |
| $e_{i,j}$ | scalar | Alignment score |

Where $d_a$ is the alignment model's hidden dimension.

**Computation breakdown:**

$$\underbrace{W_a \cdot s_{i-1}}_{(d_a \times 1)} + \underbrace{U_a \cdot h_j}_{(d_a \times 1)} + \underbrace{b_a}_{(d_a \times 1)} \xrightarrow{\tanh} \underbrace{(d_a \times 1)}_{} \xrightarrow{v_a^T \cdot} \underbrace{e_{i,j}}_{\text{scalar}}$$

### Attention Weights

$$\boxed{\alpha_{i,j} = \frac{\exp(e_{i,j})}{\sum_{k=1}^{T} \exp(e_{i,k})}} \quad \text{where} \quad \sum_{j=1}^{T} \alpha_{i,j} = 1$$

### Context Vector

$$\boxed{c_i = \sum_{j=1}^{T} \alpha_{i,j} \cdot h_j \quad \in \mathbb{R}^{2n_h}}$$

A **different** context vector is computed for **each decoder step** $i$.

### Decoder Equations

The decoder is a unidirectional RNN (LSTM or GRU) that receives the context vector at each step:

$$\boxed{s_i = f(s_{i-1}, \; y_{i-1}, \; c_i)}$$

For an LSTM decoder, this expands to the full gate equations where the input is the concatenation $[y_{i-1} \; ; \; c_i]$:

$$\begin{aligned}
f_i^{\text{dec}} &= \sigma(W_f^{\text{dec}} \cdot [s_{i-1}, \; y_{i-1}, \; c_i] + b_f^{\text{dec}}) \\[4pt]
i_i^{\text{dec}} &= \sigma(W_i^{\text{dec}} \cdot [s_{i-1}, \; y_{i-1}, \; c_i] + b_i^{\text{dec}}) \\[4pt]
\tilde{C}_i^{\text{dec}} &= \tanh(W_C^{\text{dec}} \cdot [s_{i-1}, \; y_{i-1}, \; c_i] + b_C^{\text{dec}}) \\[4pt]
C_i^{\text{dec}} &= f_i^{\text{dec}} \odot C_{i-1}^{\text{dec}} + i_i^{\text{dec}} \odot \tilde{C}_i^{\text{dec}} \\[4pt]
o_i^{\text{dec}} &= \sigma(W_o^{\text{dec}} \cdot [s_{i-1}, \; y_{i-1}, \; c_i] + b_o^{\text{dec}}) \\[4pt]
s_i &= o_i^{\text{dec}} \odot \tanh(C_i^{\text{dec}})
\end{aligned}$$

### Output Prediction

$$\boxed{\hat{y}_i = \text{softmax}(W_y \cdot [s_i \; ; \; c_i] + b_y)}$$

> The output prediction uses **both** the decoder state $s_i$ and the context vector $c_i$. This gives the model two pathways of information: what it has decoded so far ($s_i$) and what the input says is relevant right now ($c_i$).

---

## 6. Types of Attention

### Bahdanau Attention (Additive)

Proposed in **"Neural Machine Translation by Jointly Learning to Align and Translate"** (Bahdanau, Cho, Bengio, 2014).

$$\boxed{e_{i,j} = v_a^T \cdot \tanh(W_a \cdot s_{i-1} + U_a \cdot h_j)}$$

Key characteristics:
- Uses the **previous** decoder state $s_{i-1}$ (before the current step)
- Alignment is computed via **addition** of projected states, followed by $\tanh$ and a linear projection
- The alignment model is a small **feed-forward neural network** (hence "additive")
- Context vector is used as **input to the decoder RNN** at the current step

**Data flow at decoder step $i$:**

$$s_{i-1} \xrightarrow{\text{alignment}} e_{i,j} \xrightarrow{\text{softmax}} \alpha_{i,j} \xrightarrow{\text{weighted sum}} c_i \xrightarrow{\text{+ } y_{i-1}} s_i \xrightarrow{\text{predict}} \hat{y}_i$$

### Luong Attention (Multiplicative)

Proposed in **"Effective Approaches to Attention-based Neural Machine Translation"** (Luong, Pham, Manning, 2015).

$$\boxed{e_{i,j} = s_i^T \cdot W_a \cdot h_j \quad \text{(General)}}$$

$$\boxed{e_{i,j} = s_i^T \cdot h_j \quad \text{(Dot Product)}}$$

Key characteristics:
- Uses the **current** decoder state $s_i$ (after the current step)
- Alignment is computed via **dot product** or bilinear form (hence "multiplicative")
- Computationally simpler and faster
- Context vector is used to **predict the output** (not as input to the RNN)

**Data flow at decoder step $i$:**

$$s_{i-1} \xrightarrow{\text{RNN}} s_i \xrightarrow{\text{alignment}} e_{i,j} \xrightarrow{\text{softmax}} \alpha_{i,j} \xrightarrow{\text{weighted sum}} c_i \xrightarrow{[s_i \; ; \; c_i]} \hat{y}_i$$

Luong also proposed three scoring functions:

| Name | Formula | Parameters |
|---|---|---|
| **Dot** | $e_{i,j} = s_i^T \cdot h_j$ | None (parameter-free) |
| **General** | $e_{i,j} = s_i^T \cdot W_a \cdot h_j$ | $W_a \in \mathbb{R}^{n_s \times 2n_h}$ |
| **Concat** | $e_{i,j} = v_a^T \cdot \tanh(W_a \cdot [s_i \; ; \; h_j])$ | $W_a, v_a$ (same as Bahdanau) |

### Comparison Table

| Feature | Bahdanau (Additive) | Luong (Multiplicative) |
|---|---|---|
| **Paper** | 2014 | 2015 |
| **Decoder state used** | $s_{i-1}$ (previous) | $s_i$ (current) |
| **Score function** | $v_a^T \tanh(Ws + Uh)$ | $s^T W h$ or $s^T h$ |
| **Computation** | More expensive (MLP) | Cheaper (dot product) |
| **Context vector used in** | Decoder RNN input | Output prediction |
| **Encoder** | Bidirectional | Unidirectional (top layer) |

---

## 7. Full Walkthrough: "Hello What's Up" to "Gracias"

Let's trace the complete Attention Mechanism for translating the input **"Hello What's Up"** into French.

### Phase 1 — Encoder (Bidirectional RNN)

**Input:** $(x_1, x_2, x_3)$ = ("Hello", "What's", "Up")

**Forward LSTM:**

| $t$ | Input | State |
|---|---|---|
| 1 | "Hello" | $\overrightarrow{h_1}$: context from "Hello" |
| 2 | "What's" | $\overrightarrow{h_2}$: context from "Hello What's" |
| 3 | "Up" | $\overrightarrow{h_3}$: context from "Hello What's Up" |

**Backward LSTM:**

| $t$ | Input | State |
|---|---|---|
| 3 | "Up" | $\overleftarrow{h_3}$: context from "Up" |
| 2 | "What's" | $\overleftarrow{h_2}$: context from "What's Up" |
| 1 | "Hello" | $\overleftarrow{h_1}$: context from "Hello What's Up" |

**Annotations (concatenation):**

$$h_1 = [\overrightarrow{h_1} ; \overleftarrow{h_1}] \quad \text{— centered on "Hello", knows full sentence}$$

$$h_2 = [\overrightarrow{h_2} ; \overleftarrow{h_2}] \quad \text{— centered on "What's", knows full sentence}$$

$$h_3 = [\overrightarrow{h_3} ; \overleftarrow{h_3}] \quad \text{— centered on "Up", knows full sentence}$$

The initial decoder state $s_0$ is typically set from the encoder's final states.

### Phase 2 — Decoder Step 1 (Generate $\hat{y}_1$)

**Goal:** Generate the first output word.

**Step 1 — Alignment scores:**

$$e_{1,1} = a(s_0, h_1) \quad \text{— How relevant is "Hello"?}$$

$$e_{1,2} = a(s_0, h_2) \quad \text{— How relevant is "What's"?}$$

$$e_{1,3} = a(s_0, h_3) \quad \text{— How relevant is "Up"?}$$

Suppose: $[e_{1,1}, e_{1,2}, e_{1,3}] = [3.2, 1.1, 0.8]$

**Step 2 — Attention weights (softmax):**

$$[\alpha_{1,1}, \alpha_{1,2}, \alpha_{1,3}] = \text{softmax}([3.2, 1.1, 0.8]) = [\mathbf{0.78}, 0.12, 0.10]$$

> The decoder is paying **78% attention to "Hello"** for generating the first word.

**Step 3 — Context vector:**

$$c_1 = 0.78 \cdot h_1 + 0.12 \cdot h_2 + 0.10 \cdot h_3$$

$c_1$ is heavily influenced by $h_1$ ("Hello").

**Step 4 — Decoder update and predict:**

$$s_1 = \text{LSTM}(s_0, \; y_0 = \text{Embed(SOS)}, \; c_1)$$

$$\hat{y}_1 = \text{softmax}(W_y \cdot [s_1 ; c_1] + b_y) \rightarrow \textbf{"Salut"}$$

### Phase 3 — Decoder Step 2 (Generate $\hat{y}_2$)

Now the attention **shifts** to different parts of the input.

**Step 1 — New alignment scores:**

$$[e_{2,1}, e_{2,2}, e_{2,3}] = [0.4, 2.8, 1.5]$$

**Step 2 — New attention weights:**

$$[\alpha_{2,1}, \alpha_{2,2}, \alpha_{2,3}] = \text{softmax}([0.4, 2.8, 1.5]) = [0.06, \mathbf{0.72}, 0.22]$$

> Now the decoder pays **72% attention to "What's"** — it has shifted focus.

**Step 3 — New context vector:**

$$c_2 = 0.06 \cdot h_1 + 0.72 \cdot h_2 + 0.22 \cdot h_3$$

**Step 4 — Predict:**

$$s_2 = \text{LSTM}(s_1, \; y_1 = \text{Embed("Salut")}, \; c_2)$$

$$\hat{y}_2 = \text{softmax}(W_y \cdot [s_2 ; c_2] + b_y) \rightarrow \textbf{"quoi"}$$

### Phase 4 — Decoder Step 3 (Generate $\hat{y}_3$)

**Attention weights:** $[\alpha_{3,1}, \alpha_{3,2}, \alpha_{3,3}] = [0.05, 0.15, \mathbf{0.80}]$

> Now **80% attention on "Up"**.

$$c_3 = 0.05 \cdot h_1 + 0.15 \cdot h_2 + 0.80 \cdot h_3$$

$$\hat{y}_3 = \text{softmax}(W_y \cdot [s_3 ; c_3] + b_y) \rightarrow \textbf{"de neuf"}$$

### Phase 5 — Decoder Step 4

$$\hat{y}_4 \rightarrow \textbf{\<EOS\>} \quad \text{(STOP)}$$

**Final translation:** "Salut quoi de neuf"

> Notice how the attention **dynamically shifted** across decoder steps: first focusing on "Hello", then "What's", then "Up". Each output word got a **custom view** of the input.

---

## 8. Visualizing Attention: The Alignment Matrix

The attention weights across all decoder steps form an **alignment matrix** $A$ where $A_{i,j} = \alpha_{i,j}$:

```
              Encoder (Input)
              Hello  What's   Up
            +------+--------+------+
  Salut     | 0.78 |  0.12  | 0.10 |  <-- Step 1: Focus on "Hello"
            +------+--------+------+
Decoder     |      |        |      |
  quoi      | 0.06 |  0.72  | 0.22 |  <-- Step 2: Focus on "What's"
(Output)    +------+--------+------+
            |      |        |      |
  de neuf   | 0.05 |  0.15  | 0.80 |  <-- Step 3: Focus on "Up"
            +------+--------+------+
```

This matrix is often visualized as a **heatmap**. For a well-trained translation model, you typically see a roughly **diagonal pattern** (input word $j$ aligns with output word $i$), but with interesting deviations where word order differs between languages.

**What the alignment matrix reveals:**
- **Diagonal pattern:** Languages with similar word order (English to French)
- **Crossed lines:** Languages where word order reverses (English adjective-noun vs French noun-adjective)
- **One-to-many:** One input word maps to multiple output words
- **Many-to-one:** Multiple input words compress into one output word

---

## 9. Training with Attention

### Loss Function

Same cross-entropy loss as basic Seq2Seq, applied at each decoder step:

$$\boxed{\mathcal{L} = -\sum_{i=1}^{n} \log P(y_i^{\text{true}} | y_{<i}, \mathbf{x})}$$

$$= -\sum_{i=1}^{n} \log \hat{y}_{i, \; w_i^{\text{true}}}$$

Where $\hat{y}_{i, \; w_i^{\text{true}}}$ is the probability assigned to the correct word at step $i$.

The attention weights $\alpha_{i,j}$ are **not directly supervised** — there is no explicit loss for "attend to the right place." Instead, the attention learns correct alignments **implicitly** through the translation loss. If attending to the wrong input position leads to a wrong output word, the loss increases, and backpropagation adjusts the alignment model parameters.

### Gradient Flow

Gradients flow through **three paths** back to the encoder:

**Path 1 — Through decoder states:**

$$\mathcal{L} \xrightarrow{\nabla} s_i \xrightarrow{\nabla} s_{i-1} \xrightarrow{\nabla} \cdots \quad \text{(standard BPTT in decoder)}$$

**Path 2 — Through context vectors (the attention shortcut):**

$$\mathcal{L} \xrightarrow{\nabla} c_i \xrightarrow{\nabla} \alpha_{i,j} \cdot h_j \xrightarrow{\nabla} h_j \quad \text{(direct gradient to encoder position } j\text{)}$$

**Path 3 — Through alignment scores:**

$$\mathcal{L} \xrightarrow{\nabla} \alpha_{i,j} \xrightarrow{\nabla} e_{i,j} \xrightarrow{\nabla} W_a, U_a, v_a \quad \text{(updates alignment model)}$$

> **Path 2 is crucial:** In basic Seq2Seq, encoder gradients must flow through the entire decoder chain (vanishing gradient risk). With attention, gradients can flow **directly** from the loss to any encoder position $h_j$ through the context vector. This creates a **gradient shortcut** that makes training much more effective, especially for long sequences.

---

## 10. Why Attention Solves the Bottleneck

| Problem in Basic Seq2Seq | How Attention Solves It |
|---|---|
| **Single fixed context vector** | A **different context vector** $c_i$ is computed at each decoder step |
| **All info squeezed into one vector** | Decoder can access **all** $T$ encoder states directly |
| **Early tokens forgotten** | Attention weights let the decoder **reach back** to any position |
| **Same context for every output** | Context is **dynamically tailored** to each output token |
| **BLEU decays for long sentences** | Direct access to all positions **maintains quality** regardless of length |
| **Vanishing gradients to encoder** | **Gradient shortcuts** through attention weights |

**The key insight in one equation:**

Basic Seq2Seq:

$$\text{Decoder sees:} \quad h_T \quad \text{(1 vector, regardless of input length)}$$

Attention:

$$\text{Decoder sees:} \quad c_i = \sum_{j=1}^{T} \alpha_{i,j} \cdot h_j \quad \text{(weighted access to ALL } T \text{ vectors, different at each step)}$$

---

## 11. Limitations of Attention in RNNs

Despite solving the bottleneck, attention-based Seq2Seq still has limitations:

| Limitation | Description |
|---|---|
| **Sequential computation** | The encoder and decoder RNNs still process tokens **one at a time**. Cannot parallelize across time steps. |
| **$O(T \times n)$ attention cost** | At each of $n$ decoder steps, attention computes scores over all $T$ encoder positions. For long sequences, this is expensive. |
| **RNN memory limits** | Even with attention, the decoder hidden state $s_i$ is still a fixed-size recurrent state that can struggle with very long outputs. |
| **Training speed** | Sequential RNN processing cannot leverage GPU parallelism as effectively as fully parallel architectures. |
| **No self-attention** | The encoder tokens don't attend to each other — each position only sees its local bidirectional context, not learned global relationships. |

---

## 12. From Attention to Transformers (Preview)

The Transformer architecture (Vaswani et al., 2017, "Attention Is All You Need") takes the attention idea to its logical conclusion:

$$\text{RNN + Attention} \xrightarrow{\text{remove RNN entirely}} \text{Self-Attention (Transformer)}$$

| Feature | RNN + Attention | Transformer |
|---|---|---|
| Sequence processing | Sequential (RNN) | **Fully parallel** (self-attention) |
| Encoder attention | Cross-attention (decoder $\rightarrow$ encoder) | Cross-attention **+ self-attention** |
| Positional info | Inherent in RNN recurrence | Explicit **positional encodings** |
| Attention type | Decoder attends to encoder | **Self-attention** (every token attends to every other token) |
| Training speed | Slow (sequential) | **Fast** (parallelizable) |

> The Transformer keeps the attention mechanism (the best part of this architecture) and removes the RNN (the bottleneck for parallelization). This is the foundation of modern models like BERT, GPT, and all large language models.

---

## 13. Summary

### The Problem

$$\text{Basic Seq2Seq:} \quad \text{Entire input} \xrightarrow{\text{compress}} \underbrace{h_T}_{\text{single vector}} \xrightarrow{\text{decode}} \text{Output (quality degrades with length)}$$

### The Solution — Attention

$$\boxed{\begin{aligned}
&\text{1. Annotations:} & h_j &= [\overrightarrow{h_j} ; \overleftarrow{h_j}] && \text{Bidirectional encoder states} \\[8pt]
&\text{2. Alignment:} & e_{i,j} &= a(s_{i-1}, h_j) && \text{Score relevance of input } j \text{ for output } i \\[8pt]
&\text{3. Weights:} & \alpha_{i,j} &= \text{softmax}_j(e_{i,j}) && \text{Normalize to probability distribution} \\[8pt]
&\text{4. Context:} & c_i &= \textstyle\sum_{j=1}^{T} \alpha_{i,j} \cdot h_j && \text{Weighted sum — custom for each step} \\[8pt]
&\text{5. Decode:} & s_i &= f(s_{i-1}, y_{i-1}, c_i) && \text{Decoder update with attention context} \\[8pt]
&\text{6. Output:} & \hat{y}_i &= \text{softmax}(W_y \cdot [s_i ; c_i]) && \text{Predict next token}
\end{aligned}}$$

### Core Insight

At every decoder step, attention asks: **"Which parts of the input are most relevant right now?"** and constructs a custom-weighted view of the entire input to answer that question.

$$c_i = \underbrace{\alpha_{i,1}}_{\text{weight}} \cdot \underbrace{h_1}_{\text{"Hello"}} + \underbrace{\alpha_{i,2}}_{\text{weight}} \cdot \underbrace{h_2}_{\text{"What's"}} + \underbrace{\alpha_{i,3}}_{\text{weight}} \cdot \underbrace{h_3}_{\text{"Up"}}$$

### The Evolution

$$\text{Seq2Seq} \xrightarrow[\text{bottleneck}]{\text{fixed context}} \text{Attention} \xrightarrow[\text{sequential RNN}]{\text{remove}} \text{Transformer} \xrightarrow[\text{scale up}]{\text{pre-train}} \text{BERT, GPT, LLMs}$$
