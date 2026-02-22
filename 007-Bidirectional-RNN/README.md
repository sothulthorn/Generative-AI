# Bidirectional RNN - In-Depth Intuition

## Table of Contents

1. [Recap: Unidirectional RNN](#1-recap-unidirectional-rnn)
   - [Simple RNN, LSTM, GRU](#simple-rnn-lstm-gru)
   - [Types of RNN](#types-of-rnn)
2. [The Problem with Unidirectional RNNs](#2-the-problem-with-unidirectional-rnns)
   - [Left-to-Right Limitation](#left-to-right-limitation)
   - [The "Dosa vs Pizza" Example](#the-dosa-vs-pizza-example)
3. [Why Bidirectional RNN?](#3-why-bidirectional-rnn)
4. [Bidirectional RNN Architecture](#4-bidirectional-rnn-architecture)
   - [Two Parallel Layers](#two-parallel-layers)
   - [Forward Layer](#forward-layer)
   - [Backward Layer](#backward-layer)
   - [Combining Outputs](#combining-outputs)
5. [Mathematical Formulation](#5-mathematical-formulation)
   - [Forward Pass Equations](#forward-pass-equations)
   - [Output Computation](#output-computation)
   - [Dimension Analysis](#dimension-analysis)
6. [Step-by-Step Walkthrough](#6-step-by-step-walkthrough)
7. [Backpropagation in Bidirectional RNN](#7-backpropagation-in-bidirectional-rnn)
8. [Bidirectional LSTM & GRU](#8-bidirectional-lstm--gru)
   - [Bidirectional LSTM](#bidirectional-lstm)
   - [Bidirectional GRU](#bidirectional-gru)
9. [Applications in NLP](#9-applications-in-nlp)
   - [Named Entity Recognition (NER)](#named-entity-recognition-ner)
   - [Part-of-Speech Tagging (POS)](#part-of-speech-tagging-pos)
   - [Sentiment Analysis](#sentiment-analysis)
   - [Machine Translation](#machine-translation)
   - [Question Answering](#question-answering)
   - [Text Classification](#text-classification)
10. [Limitations](#10-limitations)
11. [Summary](#11-summary)

---

## 1. Recap: Unidirectional RNN

### Simple RNN, LSTM, GRU

Before understanding the Bidirectional RNN, let's recall the progression:

$$\text{Simple RNN} \rightarrow \text{LSTM RNN} \rightarrow \text{GRU RNN}$$

All three of these are **unidirectional** — they process the input sequence in **one direction only** (left to right, i.e., from $t = 1$ to $t = T$).

At each time step $t$, a unidirectional RNN computes:

$$h_t = f(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)$$

$$\hat{y}_t = g(W_{hy} \cdot h_t + b_y)$$

The hidden state $h_t$ at any time step only carries information from **past inputs** $(x_1, x_2, \ldots, x_t)$. It has **no knowledge of future inputs** $(x_{t+1}, x_{t+2}, \ldots, x_T)$.

### Types of RNN

| Architecture | Input $\rightarrow$ Output | Example |
|---|---|---|
| **One-to-One** | Single input $\rightarrow$ Single output | Standard classification |
| **One-to-Many** | Single input $\rightarrow$ Sequence output | Image captioning |
| **Many-to-One** | Sequence input $\rightarrow$ Single output | Sentiment analysis |
| **Many-to-Many** | Sequence input $\rightarrow$ Sequence output | Machine translation, NER |

All of these architectures can be built with either unidirectional or bidirectional RNNs. The bidirectional variant is especially powerful for **many-to-many** tasks where context from both directions matters.

---

## 2. The Problem with Unidirectional RNNs

### Left-to-Right Limitation

In a standard (unidirectional) RNN, the hidden state $h_t$ at time step $t$ is computed **only** from past inputs:

$$h_t = f(x_1, x_2, \ldots, x_t)$$

This means when the network makes a prediction at position $t$, it can only use information from **the left side** (previous tokens). It is completely **blind to what comes after** position $t$.

For many NLP tasks, **future context is just as important as past context**.

### The "Dosa vs Pizza" Example

Consider the sentence with a blank to fill:

> **"Krish eats ___ in Bangalore"**

A unidirectional RNN processes this left-to-right:

| Time Step | Token | Hidden State Knows |
|---|---|---|
| $t = 1$ | Krish | "Krish" |
| $t = 2$ | eats | "Krish eats" |
| $t = 3$ | ___ | "Krish eats" — must predict here |
| $t = 4$ | in | "Krish eats ___ in" |
| $t = 5$ | Bangalore | "Krish eats ___ in Bangalore" |

At $t = 3$, the model must predict the blank. But it only knows **"Krish eats"** — it has **no idea** that "Bangalore" comes later. Without future context, the model might predict any food: pizza, sushi, dosa, etc.

Now consider what happens if the model also had access to the **future context**:

| Sentence | Future Context | Prediction |
|---|---|---|
| "Krish eats ___ in **Bangalore**" | Bangalore (a city in India) | **Dosa** (South Indian dish) |
| "Krish eats ___ in **Paris**" | Paris (a city in France) | **Pizza**, Croissant, etc. |

The **future word "Bangalore" or "Paris"** dramatically changes what the correct prediction should be. A unidirectional RNN cannot capture this because it hasn't seen those words yet at the time of prediction.

> This is the fundamental motivation for Bidirectional RNNs: many predictions require **both past and future context**.

---

## 3. Why Bidirectional RNN?

A Bidirectional RNN solves the limitation by processing the sequence in **both directions simultaneously**:

$$\text{Forward:} \quad x_1 \rightarrow x_2 \rightarrow x_3 \rightarrow \cdots \rightarrow x_T$$

$$\text{Backward:} \quad x_T \rightarrow x_{T-1} \rightarrow x_{T-2} \rightarrow \cdots \rightarrow x_1$$

At each time step $t$, the network has access to **the entire sequence context** — both what came before and what comes after.

| Feature | Unidirectional RNN | Bidirectional RNN |
|---|---|---|
| Direction | Left-to-right only ($\rightarrow$) | Left-to-right ($\rightarrow$) **and** right-to-left ($\leftarrow$) |
| Context at step $t$ | Past only: $(x_1, \ldots, x_t)$ | Full: $(x_1, \ldots, x_T)$ |
| Hidden state | Single $h_t$ | Two: $\overrightarrow{h_t}$ (forward) + $\overleftarrow{h_t}$ (backward) |
| Parameters | $\sim N$ | $\sim 2N$ (roughly double) |
| Suitable for | Real-time / streaming tasks | Tasks where full sequence is available |

---

## 4. Bidirectional RNN Architecture

### Two Parallel Layers

The Bidirectional RNN consists of **two independent RNN layers** that process the same input sequence in opposite directions:

```
        y_1         y_2         y_3         y_4
         ^           ^           ^           ^
         |           |           |           |
      [Combine]  [Combine]  [Combine]  [Combine]
       /    \     /    \     /    \     /    \
      /      \   /      \   /      \   /      \
  -->[ h1 ]-->[ h2 ]-->[ h3 ]-->[ h4 ]-->       Forward Layer
      |        |        |        |
      |        |        |        |
  <--[ h1 ]<--[ h2 ]<--[ h3 ]<--[ h4 ]<--       Backward Layer
      \      /   \      /   \      /   \      /
       \    /     \    /     \    /     \    /
        v           v           v           v
       x_1        x_2        x_3        x_4
```

Both layers receive the **same input** at each time step, but they propagate information in opposite directions.

### Forward Layer

The forward layer processes the input from $t = 1$ to $t = T$ (left to right):

$$\overrightarrow{h_t} = f\left(W_{\overrightarrow{h}} \cdot \overrightarrow{h_{t-1}} + W_{\overrightarrow{x}} \cdot x_t + b_{\overrightarrow{h}}\right)$$

At each time step, the forward hidden state $\overrightarrow{h_t}$ captures information from **all past inputs** $(x_1, x_2, \ldots, x_t)$.

| Time Step | Forward State Contains |
|---|---|
| $\overrightarrow{h_1}$ | Context from $x_1$ |
| $\overrightarrow{h_2}$ | Context from $x_1, x_2$ |
| $\overrightarrow{h_3}$ | Context from $x_1, x_2, x_3$ |
| $\overrightarrow{h_T}$ | Context from $x_1, x_2, \ldots, x_T$ (entire sequence) |

### Backward Layer

The backward layer processes the input from $t = T$ to $t = 1$ (right to left):

$$\overleftarrow{h_t} = f\left(W_{\overleftarrow{h}} \cdot \overleftarrow{h_{t+1}} + W_{\overleftarrow{x}} \cdot x_t + b_{\overleftarrow{h}}\right)$$

At each time step, the backward hidden state $\overleftarrow{h_t}$ captures information from **all future inputs** $(x_t, x_{t+1}, \ldots, x_T)$.

| Time Step | Backward State Contains |
|---|---|
| $\overleftarrow{h_T}$ | Context from $x_T$ |
| $\overleftarrow{h_{T-1}}$ | Context from $x_{T-1}, x_T$ |
| $\overleftarrow{h_{T-2}}$ | Context from $x_{T-2}, x_{T-1}, x_T$ |
| $\overleftarrow{h_1}$ | Context from $x_1, x_2, \ldots, x_T$ (entire sequence) |

### Combining Outputs

At each time step $t$, the forward and backward hidden states are **combined** to form a single representation that captures the **full bidirectional context**:

$$h_t = [\overrightarrow{h_t} ; \overleftarrow{h_t}]$$

The most common combination method is **concatenation**, but other strategies exist:

| Method | Formula | Output Dimension |
|---|---|---|
| **Concatenation** (most common) | $h_t = [\overrightarrow{h_t} ; \overleftarrow{h_t}]$ | $2n_h$ |
| **Summation** | $h_t = \overrightarrow{h_t} + \overleftarrow{h_t}$ | $n_h$ |
| **Averaging** | $h_t = \frac{\overrightarrow{h_t} + \overleftarrow{h_t}}{2}$ | $n_h$ |
| **Element-wise multiplication** | $h_t = \overrightarrow{h_t} \odot \overleftarrow{h_t}$ | $n_h$ |

Where $n_h$ is the hidden state dimension of each individual layer.

Concatenation is preferred because it **preserves all information** from both directions without any lossy compression.

---

## 5. Mathematical Formulation

### Forward Pass Equations

Given an input sequence $(x_1, x_2, \ldots, x_T)$, the Bidirectional RNN computes:

**Forward Layer** (processes $t = 1, 2, \ldots, T$):

$$\boxed{\overrightarrow{h_t} = \tanh\left(W_{\overrightarrow{h}} \cdot \overrightarrow{h_{t-1}} + W_{\overrightarrow{x}} \cdot x_t + b_{\overrightarrow{h}}\right)}$$

**Backward Layer** (processes $t = T, T-1, \ldots, 1$):

$$\boxed{\overleftarrow{h_t} = \tanh\left(W_{\overleftarrow{h}} \cdot \overleftarrow{h_{t+1}} + W_{\overleftarrow{x}} \cdot x_t + b_{\overleftarrow{h}}\right)}$$

**Combined Hidden State:**

$$\boxed{h_t = [\overrightarrow{h_t} ; \overleftarrow{h_t}]}$$

### Output Computation

The prediction at each time step uses the combined hidden state:

$$\boxed{\hat{y}_t = g\left(W_y \cdot h_t + b_y\right) = g\left(W_y \cdot [\overrightarrow{h_t} ; \overleftarrow{h_t}] + b_y\right)}$$

Where $g$ is the output activation function (e.g., softmax for classification, sigmoid for binary tasks).

### Dimension Analysis

Assume:
- Input dimension: $n_x$
- Hidden state dimension (per direction): $n_h$
- Output dimension: $n_y$

| Parameter | Shape | Layer |
|---|---|---|
| $W_{\overrightarrow{x}}$ | $(n_x \times n_h)$ | Forward: input-to-hidden |
| $W_{\overrightarrow{h}}$ | $(n_h \times n_h)$ | Forward: hidden-to-hidden |
| $b_{\overrightarrow{h}}$ | $(1 \times n_h)$ | Forward: bias |
| $W_{\overleftarrow{x}}$ | $(n_x \times n_h)$ | Backward: input-to-hidden |
| $W_{\overleftarrow{h}}$ | $(n_h \times n_h)$ | Backward: hidden-to-hidden |
| $b_{\overleftarrow{h}}$ | $(1 \times n_h)$ | Backward: bias |
| $W_y$ | $(2n_h \times n_y)$ | Output layer (takes concatenated state) |
| $b_y$ | $(1 \times n_y)$ | Output: bias |

**Total recurrent parameters:** $2 \times [n_h \times (n_x + n_h) + n_h]$

> A Bidirectional RNN has roughly **2x the recurrent parameters** of a unidirectional RNN — one complete set of weights for each direction. The output layer weight matrix $W_y$ is also wider ($2n_h$ instead of $n_h$) because it receives the concatenated hidden state.

### All Parameters at a Glance

$$\boxed{\begin{aligned}
\overrightarrow{h_t} &= \tanh(W_{\overrightarrow{h}} \cdot \overrightarrow{h_{t-1}} + W_{\overrightarrow{x}} \cdot x_t + b_{\overrightarrow{h}}) && \text{(1) Forward Hidden State} \\[8pt]
\overleftarrow{h_t} &= \tanh(W_{\overleftarrow{h}} \cdot \overleftarrow{h_{t+1}} + W_{\overleftarrow{x}} \cdot x_t + b_{\overleftarrow{h}}) && \text{(2) Backward Hidden State} \\[8pt]
h_t &= [\overrightarrow{h_t} ; \overleftarrow{h_t}] && \text{(3) Combined State} \\[8pt]
\hat{y}_t &= g(W_y \cdot h_t + b_y) && \text{(4) Output}
\end{aligned}}$$

---

## 6. Step-by-Step Walkthrough

Let's trace through the "Krish eats ___ in Bangalore" example with a Bidirectional RNN.

**Input sequence:** $(x_1, x_2, x_3, x_4, x_5)$ = ("Krish", "eats", "___", "in", "Bangalore")

### Step 1 — Forward Layer (Left to Right)

$$\overrightarrow{h_0} = \mathbf{0} \quad \text{(initial hidden state)}$$

| Step | Input | Computation | State Captures |
|---|---|---|---|
| $t=1$ | "Krish" | $\overrightarrow{h_1} = \tanh(W_{\overrightarrow{h}} \cdot \overrightarrow{h_0} + W_{\overrightarrow{x}} \cdot x_1 + b)$ | "Krish" |
| $t=2$ | "eats" | $\overrightarrow{h_2} = \tanh(W_{\overrightarrow{h}} \cdot \overrightarrow{h_1} + W_{\overrightarrow{x}} \cdot x_2 + b)$ | "Krish eats" |
| $t=3$ | "___" | $\overrightarrow{h_3} = \tanh(W_{\overrightarrow{h}} \cdot \overrightarrow{h_2} + W_{\overrightarrow{x}} \cdot x_3 + b)$ | "Krish eats ___" |
| $t=4$ | "in" | $\overrightarrow{h_4} = \tanh(W_{\overrightarrow{h}} \cdot \overrightarrow{h_3} + W_{\overrightarrow{x}} \cdot x_4 + b)$ | "Krish eats ___ in" |
| $t=5$ | "Bangalore" | $\overrightarrow{h_5} = \tanh(W_{\overrightarrow{h}} \cdot \overrightarrow{h_4} + W_{\overrightarrow{x}} \cdot x_5 + b)$ | "Krish eats ___ in Bangalore" |

### Step 2 — Backward Layer (Right to Left)

$$\overleftarrow{h_6} = \mathbf{0} \quad \text{(initial hidden state)}$$

| Step | Input | Computation | State Captures |
|---|---|---|---|
| $t=5$ | "Bangalore" | $\overleftarrow{h_5} = \tanh(W_{\overleftarrow{h}} \cdot \overleftarrow{h_6} + W_{\overleftarrow{x}} \cdot x_5 + b)$ | "Bangalore" |
| $t=4$ | "in" | $\overleftarrow{h_4} = \tanh(W_{\overleftarrow{h}} \cdot \overleftarrow{h_5} + W_{\overleftarrow{x}} \cdot x_4 + b)$ | "in Bangalore" |
| $t=3$ | "___" | $\overleftarrow{h_3} = \tanh(W_{\overleftarrow{h}} \cdot \overleftarrow{h_4} + W_{\overleftarrow{x}} \cdot x_3 + b)$ | "___ in Bangalore" |
| $t=2$ | "eats" | $\overleftarrow{h_2} = \tanh(W_{\overleftarrow{h}} \cdot \overleftarrow{h_3} + W_{\overleftarrow{x}} \cdot x_2 + b)$ | "eats ___ in Bangalore" |
| $t=1$ | "Krish" | $\overleftarrow{h_1} = \tanh(W_{\overleftarrow{h}} \cdot \overleftarrow{h_2} + W_{\overleftarrow{x}} \cdot x_1 + b)$ | "Krish eats ___ in Bangalore" |

### Step 3 — Combine and Predict

At the blank position ($t = 3$):

$\overrightarrow{h_3}$ captures: **"Krish eats ___"** (past context)

$\overleftarrow{h_3}$ captures: **"___ in Bangalore"** (future context)

$$h_3 = [\overrightarrow{h_3} ; \overleftarrow{h_3}]$$

$$\hat{y}_3 = \text{softmax}(W_y \cdot h_3 + b_y)$$

Now the model sees **both** "Krish eats" and "in Bangalore" when predicting the blank — so it can confidently predict **"Dosa"** (a South Indian dish, fitting the Bangalore context).

> If the sentence were "Krish eats ___ in **Paris**", the backward layer would carry "in Paris" to position 3, and the model would predict something like **"Pizza"** or **"Croissant"** instead.

---

## 7. Backpropagation in Bidirectional RNN

Training a Bidirectional RNN uses **Backpropagation Through Time (BPTT)** applied to both layers independently.

### Loss Function

The total loss over the sequence:

$$\mathcal{L} = \sum_{t=1}^{T} \ell(\hat{y}_t, y_t)$$

### Gradient Flow

The gradients flow through **two separate paths**:

**Path 1 — Forward layer gradients:**

$$\frac{\partial \mathcal{L}}{\partial W_{\overrightarrow{h}}} = \sum_{t=1}^{T} \frac{\partial \ell_t}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial \overrightarrow{h_t}} \cdot \frac{\partial \overrightarrow{h_t}}{\partial W_{\overrightarrow{h}}}$$

Gradients propagate **backward in time** (from $t = T$ to $t = 1$) through the forward layer's hidden states.

**Path 2 — Backward layer gradients:**

$$\frac{\partial \mathcal{L}}{\partial W_{\overleftarrow{h}}} = \sum_{t=1}^{T} \frac{\partial \ell_t}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial \overleftarrow{h_t}} \cdot \frac{\partial \overleftarrow{h_t}}{\partial W_{\overleftarrow{h}}}$$

Gradients propagate **forward in time** (from $t = 1$ to $t = T$) through the backward layer's hidden states.

**Path 3 — Output layer gradients:**

$$\frac{\partial \mathcal{L}}{\partial W_y} = \sum_{t=1}^{T} \frac{\partial \ell_t}{\partial \hat{y}_t} \cdot \frac{\partial \hat{y}_t}{\partial W_y}$$

### Key Insight

The forward and backward layers are **independent** during training — gradients do not flow from one layer to the other through the hidden states. They only interact at the output layer where their hidden states are concatenated. This makes training straightforward: it's essentially training two separate RNNs that share an output layer.

---

## 8. Bidirectional LSTM & GRU

In practice, the simple RNN cells in a Bidirectional RNN are almost always replaced with **LSTM** or **GRU** cells to handle the vanishing gradient problem. The bidirectional concept is **cell-agnostic** — it works with any recurrent cell type.

### Bidirectional LSTM

Each direction uses a full LSTM cell with its own gates and cell state:

**Forward LSTM** (processes $t = 1$ to $T$):

$$\overrightarrow{f_t} = \sigma(W_{\overrightarrow{f}} \cdot [\overrightarrow{h_{t-1}}, x_t] + b_{\overrightarrow{f}})$$

$$\overrightarrow{i_t} = \sigma(W_{\overrightarrow{i}} \cdot [\overrightarrow{h_{t-1}}, x_t] + b_{\overrightarrow{i}})$$

$$\overrightarrow{\tilde{C}_t} = \tanh(W_{\overrightarrow{C}} \cdot [\overrightarrow{h_{t-1}}, x_t] + b_{\overrightarrow{C}})$$

$$\overrightarrow{C_t} = \overrightarrow{f_t} \odot \overrightarrow{C_{t-1}} + \overrightarrow{i_t} \odot \overrightarrow{\tilde{C}_t}$$

$$\overrightarrow{o_t} = \sigma(W_{\overrightarrow{o}} \cdot [\overrightarrow{h_{t-1}}, x_t] + b_{\overrightarrow{o}})$$

$$\overrightarrow{h_t} = \overrightarrow{o_t} \odot \tanh(\overrightarrow{C_t})$$

**Backward LSTM** (processes $t = T$ to $1$):

$$\overleftarrow{f_t} = \sigma(W_{\overleftarrow{f}} \cdot [\overleftarrow{h_{t+1}}, x_t] + b_{\overleftarrow{f}})$$

$$\overleftarrow{i_t} = \sigma(W_{\overleftarrow{i}} \cdot [\overleftarrow{h_{t+1}}, x_t] + b_{\overleftarrow{i}})$$

$$\overleftarrow{\tilde{C}_t} = \tanh(W_{\overleftarrow{C}} \cdot [\overleftarrow{h_{t+1}}, x_t] + b_{\overleftarrow{C}})$$

$$\overleftarrow{C_t} = \overleftarrow{f_t} \odot \overleftarrow{C_{t+1}} + \overleftarrow{i_t} \odot \overleftarrow{\tilde{C}_t}$$

$$\overleftarrow{o_t} = \sigma(W_{\overleftarrow{o}} \cdot [\overleftarrow{h_{t+1}}, x_t] + b_{\overleftarrow{o}})$$

$$\overleftarrow{h_t} = \overleftarrow{o_t} \odot \tanh(\overleftarrow{C_t})$$

**Combined output:**

$$h_t = [\overrightarrow{h_t} ; \overleftarrow{h_t}]$$

> A Bidirectional LSTM has **8 gate computations** per time step (4 forward + 4 backward), compared to 4 in a unidirectional LSTM. The total parameter count is roughly **8x** that of a simple RNN.

### Bidirectional GRU

Similarly, GRU cells can be used in both directions:

**Forward GRU:**

$$\overrightarrow{z_t} = \sigma(W_{\overrightarrow{z}} \cdot [\overrightarrow{h_{t-1}}, x_t]) \qquad \text{(Update Gate)}$$

$$\overrightarrow{r_t} = \sigma(W_{\overrightarrow{r}} \cdot [\overrightarrow{h_{t-1}}, x_t]) \qquad \text{(Reset Gate)}$$

$$\overrightarrow{\tilde{h}_t} = \tanh(W_{\overrightarrow{h}} \cdot [\overrightarrow{r_t} \odot \overrightarrow{h_{t-1}}, x_t]) \qquad \text{(Candidate)}$$

$$\overrightarrow{h_t} = (1 - \overrightarrow{z_t}) \odot \overrightarrow{h_{t-1}} + \overrightarrow{z_t} \odot \overrightarrow{\tilde{h}_t} \qquad \text{(Final State)}$$

**Backward GRU:** Same equations with $\overleftarrow{h_{t+1}}$ replacing $\overrightarrow{h_{t-1}}$.

| Variant | Gates per Direction | Total Gates (Both Directions) | Relative Parameters |
|---|---|---|---|
| Bidirectional Simple RNN | 0 | 0 | $2 \times$ Simple RNN |
| Bidirectional GRU | 2 (update, reset) | 4 | $2 \times$ GRU |
| Bidirectional LSTM | 3 (forget, input, output) + candidate | 8 | $2 \times$ LSTM |

---

## 9. Applications in NLP

Bidirectional RNNs are fundamental to many NLP tasks. Whenever the **entire input sequence is available** before predictions are made, a bidirectional model can outperform its unidirectional counterpart.

### Named Entity Recognition (NER)

**Task:** Label each word in a sentence with an entity type (Person, Location, Organization, etc.).

> **"Krish eats dosa in Bangalore"**

| Token | Forward Context | Backward Context | Combined Prediction |
|---|---|---|---|
| Krish | Start of sentence | "eats dosa in Bangalore" | **B-Person** |
| eats | "Krish" | "dosa in Bangalore" | O |
| dosa | "Krish eats" | "in Bangalore" | **B-Food** |
| in | "Krish eats dosa" | "Bangalore" | O |
| Bangalore | "Krish eats dosa in" | End of sentence | **B-Location** |

The backward context of "Bangalore" helps identify "Krish" as a person (someone who eats in a specific location). The forward context of "Krish eats" helps identify "Bangalore" as a location. **Both directions reinforce each other.**

### Part-of-Speech Tagging (POS)

**Task:** Assign a grammatical tag (noun, verb, adjective, etc.) to each word.

> **"I can can a can"**

The word "can" appears three times with three different POS tags:

| Position | Word | Forward Only | Bidirectional | Correct Tag |
|---|---|---|---|---|
| 2 | can | Ambiguous (modal verb?) | Sees "can a can" ahead → **modal verb** | **MD** (Modal) |
| 3 | can | "I can" → verb? | Sees "a can" ahead → **verb** (to can) | **VB** (Verb) |
| 5 | can | "can a" → noun? | Sees end of sentence → **noun** (a tin can) | **NN** (Noun) |

Without future context, the model struggles with such lexical ambiguity. The bidirectional model resolves it by considering the full sentence.

### Sentiment Analysis

**Task:** Determine the sentiment (positive, negative, neutral) of a text.

> **"The movie was not really all that good"**

A unidirectional model reading left-to-right might initially build up positive sentiment from "good" at the end, but miss how "not really all that" modifies it. A bidirectional model captures:
- Forward: "The movie was not really all that" → building negation context
- Backward: "good" modified by "not really all that" → negative sentiment

The combined representation captures the full nuance of negation.

### Machine Translation

**Task:** Translate a sentence from one language to another.

In the **encoder** of a sequence-to-sequence model, a Bidirectional RNN reads the entire source sentence. This is crucial because word order differs across languages:

> **English:** "The black cat" → **French:** "Le chat noir" (The cat black)

The encoder must understand that "black" modifies "cat" regardless of their relative positions. A bidirectional encoder captures this relationship from both directions.

### Question Answering

**Task:** Given a context passage and a question, identify the answer span.

> **Context:** "Albert Einstein was born in Ulm, Germany in 1879."
>
> **Question:** "Where was Einstein born?"

To identify "Ulm, Germany" as the answer, the model needs:
- Forward context: "Albert Einstein was born in" → location follows
- Backward context: "in 1879" → the answer is before the date

### Text Classification

**Task:** Classify documents into categories (spam detection, topic classification, etc.).

Bidirectional RNNs capture patterns that may appear at any position in the document. A spam indicator word at the beginning is connected to a suspicious URL at the end through the bidirectional hidden states.

---

## 10. Limitations

Despite their power, Bidirectional RNNs have important limitations:

| Limitation | Description |
|---|---|
| **Requires full sequence** | The entire input must be available before processing. Cannot be used for **real-time streaming** tasks (e.g., live speech recognition, real-time text generation). |
| **Cannot be used for generation** | Language models that generate text token-by-token (autoregressive models) cannot use future context — it doesn't exist yet. |
| **Double the computation** | Two complete RNN passes over the sequence. Training and inference are roughly **2x slower** than unidirectional. |
| **Double the parameters** | Two sets of recurrent weights. More memory required and higher risk of overfitting on small datasets. |
| **Sequential bottleneck** | Still processes tokens sequentially within each direction. Cannot fully parallelize like Transformers. |

> **When to use Bidirectional RNNs:** When the complete input is available and you need per-token predictions (NER, POS tagging) or a comprehensive sequence representation (classification, encoding for seq2seq).
>
> **When NOT to use:** Real-time streaming, autoregressive text generation, or when computational budget is extremely limited.

---

## 11. Summary

### The Problem

$$\text{Unidirectional RNN:} \quad h_t = f(x_1, x_2, \ldots, x_t) \quad \text{(blind to future)}$$

### The Solution — Bidirectional RNN

$$\boxed{\begin{aligned}
&\text{1. Forward Layer:} & \overrightarrow{h_t} &= \tanh(W_{\overrightarrow{h}} \cdot \overrightarrow{h_{t-1}} + W_{\overrightarrow{x}} \cdot x_t + b_{\overrightarrow{h}}) & &\text{→ Past context} \\[8pt]
&\text{2. Backward Layer:} & \overleftarrow{h_t} &= \tanh(W_{\overleftarrow{h}} \cdot \overleftarrow{h_{t+1}} + W_{\overleftarrow{x}} \cdot x_t + b_{\overleftarrow{h}}) & &\text{→ Future context} \\[8pt]
&\text{3. Combine:} & h_t &= [\overrightarrow{h_t} ; \overleftarrow{h_t}] & &\text{→ Full context} \\[8pt]
&\text{4. Output:} & \hat{y}_t &= g(W_y \cdot h_t + b_y) & &\text{→ Prediction}
\end{aligned}}$$

### Core Insight

$$h_t = \underbrace{[\overrightarrow{h_t}}_{\text{Past: } x_1 \ldots x_t} \; ; \; \underbrace{\overleftarrow{h_t}]}_{\text{Future: } x_t \ldots x_T}$$

> Every prediction has access to the **entire sequence** — both what came before and what comes after. This is why Bidirectional RNNs (especially Bidirectional LSTMs) became the **backbone of NLP** before the Transformer era, powering models like ELMo and forming the encoder in many seq2seq architectures.
