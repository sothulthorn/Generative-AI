# LSTM RNN - Long Short Term Memory Recurrent Neural Network

## Table of Contents

1. [Problems with RNN](#1-problems-with-rnn)
   - [The Vanishing Gradient Problem](#the-vanishing-gradient-problem)
   - [Mathematical Proof of Vanishing Gradients](#mathematical-proof-of-vanishing-gradients)
2. [Why LSTM RNN?](#2-why-lstm-rnn)
3. [How LSTM RNN Works](#3-how-lstm-rnn-works)
   - [Two Memory Channels](#two-memory-channels)
   - [Conveyor Belt Analogy](#conveyor-belt-analogy)
4. [LSTM Architecture](#4-lstm-architecture)
   - [Inputs and Outputs](#inputs-and-outputs)
   - [Key Operations](#key-operations)
   - [Vector Concatenation](#vector-concatenation)
5. [Working of LSTM RNN (Gates in Detail)](#5-working-of-lstm-rnn-gates-in-detail)
   - [5.1 Forget Gate](#51-forget-gate)
   - [5.2 Input Gate & Candidate Memory](#52-input-gate--candidate-memory)
   - [5.3 Cell State Update](#53-cell-state-update)
   - [5.4 Output Gate](#54-output-gate)
6. [Complete LSTM Forward Pass](#6-complete-lstm-forward-pass)
7. [Backpropagation & Weight Updates](#7-backpropagation--weight-updates)
8. [GRU RNN — LSTM Variant](#8-gru-rnn--lstm-variant)
9. [Summary](#9-summary)

---

## 1. Problems with RNN

A standard Recurrent Neural Network (RNN) processes sequential data by maintaining a **hidden state** that is updated at every time step. The fundamental RNN equation is:

$$h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)$$

$$\hat{y}_t = W_{hy} \cdot h_t + b_y$$

Where:
- $h_t$ is the hidden state at time step $t$
- $h_{t-1}$ is the hidden state from the previous time step
- $x_t$ is the input at time step $t$
- $W_{hh}$ is the weight matrix for hidden-to-hidden connections (recurrent weights)
- $W_{xh}$ is the weight matrix for input-to-hidden connections
- $W_{hy}$ is the weight matrix for hidden-to-output connections
- $b_h$, $b_y$ are bias vectors
- $\hat{y}_t$ is the predicted output at time step $t$

This architecture works well when the **gap** between the relevant context and the point of prediction is **small**.

### Short Gap — RNN Works Fine

> **"The color of the sky is ___"** → **blue**

Here the context words ("color", "sky") are only a few time steps away from the prediction. The hidden state $h_t$ still carries meaningful information from those nearby inputs. The RNN can propagate gradients effectively across this short distance.

### Huge Gap — RNN Fails

> **"I grew up in India ... (50 words later) ... I speak fluent ___"** → **Hindi**

The context ("India") is separated from the prediction ("Hindi") by dozens of time steps. The hidden state must carry the information about "India" through every intermediate step. In practice, this information gets **diluted and lost** because of how gradients behave during training.

### The Vanishing Gradient Problem

During training, we use **Backpropagation Through Time (BPTT)** to compute gradients. The loss $\mathcal{L}$ at time step $T$ depends on all previous hidden states. To update the weights, we need:

$$\frac{\partial \mathcal{L}}{\partial W} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}}{\partial \hat{y}_T} \cdot \frac{\partial \hat{y}_T}{\partial h_T} \cdot \frac{\partial h_T}{\partial h_t} \cdot \frac{\partial h_t}{\partial W}$$

The critical term is $\frac{\partial h_T}{\partial h_t}$, which requires the **chain rule** across all intermediate time steps:

$$\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}$$

### Mathematical Proof of Vanishing Gradients

For a standard RNN with $\tanh$ activation:

$$\frac{\partial h_k}{\partial h_{k-1}} = \text{diag}\left(\tanh'(W_{hh} \cdot h_{k-1} + W_{xh} \cdot x_k + b_h)\right) \cdot W_{hh}$$

Since $\tanh'(z) = 1 - \tanh^2(z)$ and its maximum value is **1** (at $z = 0$), and in practice it is often much less:

$$\left\| \frac{\partial h_k}{\partial h_{k-1}} \right\| \leq \| \text{diag}(\tanh') \| \cdot \| W_{hh} \|$$

When $\|W_{hh}\| < 1$ (which is common with weight values in range $[0, 1]$), the product across many time steps becomes:

$$\left\| \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}} \right\| \leq \left( \|W_{hh}\| \cdot \max|\tanh'| \right)^{T-t}$$

As $(T - t)$ grows large (huge gap), this product **exponentially decays toward zero**:

$$\prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}} \approx 0$$

**Result:** Gradients from far-away time steps vanish to nearly zero, meaning the network **cannot learn long-range dependencies**. The weights $W_{hh}$ simply stop receiving meaningful gradient updates from distant time steps.

> This is why standard RNNs fail: the chain rule, applied over many multiplication steps with small values, causes gradients to shrink exponentially.

---

## 2. Why LSTM RNN?

LSTM (Long Short Term Memory), introduced by **Hochreiter & Schmidhuber (1997)**, was specifically designed to **solve the vanishing gradient problem** and capture **long term dependencies**.

The core innovation: instead of relying on a single hidden state that gets repeatedly transformed (and loses information), LSTM introduces a **cell state** $C_t$ that acts as a **highway for information flow** — information can travel across many time steps with minimal modification.

| Feature | Standard RNN | LSTM RNN |
|---|---|---|
| Repeating module | **Single** neural network layer ($\tanh$) | **Four** interacting neural network layers |
| Memory type | Only hidden state $h_t$ (short term) | Cell state $C_t$ (long term) **+** hidden state $h_t$ (short term) |
| Information control | No gating — all info is mixed together | **Three gates** selectively control information flow |
| Gradient flow | Exponential decay (vanishing) | Protected by additive cell state updates |
| Long-range dependencies | Fails beyond ~10-20 time steps | Can handle hundreds of time steps |

**Why does LSTM solve vanishing gradients?**

The cell state update in LSTM is **additive** rather than multiplicative:

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

The gradient of $C_t$ with respect to $C_{t-1}$ is:

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

Since $f_t \in (0, 1)$ (sigmoid output) and this is a **single multiplication** (not a chain of multiplications), gradients can flow through the cell state relatively unimpeded. When $f_t \approx 1$, the gradient passes through **almost unchanged** — solving the vanishing gradient problem.

---

## 3. How LSTM RNN Works

### Basic Representation

**Standard RNN** — the repeating module contains a **single layer**:

$$h_t = \tanh(W \cdot [h_{t-1}, x_t] + b)$$

Each module takes the previous hidden state $h_{t-1}$ and current input $x_t$, passes them through a single $\tanh$ layer, and outputs the new hidden state $h_t$. Information has **only one pathway** and is forced through a squashing nonlinearity at every step.

**LSTM RNN** — the repeating module contains **four interacting layers**:

Each module has two lines running through it:
- **Top line (Cell State $C_t$):** The long term memory highway
- **Bottom line (Hidden State $h_t$):** The short term memory / working memory

The four layers are the **Forget Gate**, **Input Gate**, **Candidate Memory**, and **Output Gate** — each being a small neural network that learns to control information flow.

### Two Memory Channels

| Channel | Symbol | Role | Analogy |
|---|---|---|---|
| **Long Term Memory** | $C_t$ (Cell State) | Stores information across many time steps; modified only through controlled gate operations | A notebook where you write important facts |
| **Short Term Memory** | $h_t$ (Hidden State) | Carries the most relevant information for the immediate next prediction | Your working memory for the current task |

**Why two channels?**

- **Long Term Memory ($C_t$):** Runs along the top of the LSTM cell like a conveyor belt. Information can flow for many time steps with only minor linear interactions (pointwise multiply and add). This is what allows LSTM to remember context from 100+ steps ago.

- **Short Term Memory ($h_t$):** A filtered, compressed view of the cell state. This is what the network actually uses to make predictions and pass immediate context to the next step.

### Conveyor Belt Analogy

Think of the cell state $C_t$ as a **conveyor belt (luggage belt)** at an airport:

- Luggage (information) travels along the belt largely **unchanged**
- At certain stations, a worker (gate) can **remove** specific luggage from the belt (forget gate)
- At other stations, a worker can **place new** luggage onto the belt (input gate)
- At the end, a worker **selects** which pieces to hand to the passenger (output gate)

The belt itself moves continuously — information does not need to be repeatedly encoded and decoded, which is why it persists over long distances.

---

## 4. LSTM Architecture

### Inputs and Outputs

At each time step $t$, the LSTM cell receives **three inputs** and produces **two outputs**:

**Inputs:**

$$C_{t-1} \quad \text{— Previous cell state (Long Term Memory)}$$

$$h_{t-1} \quad \text{— Previous hidden state (Short Term Memory)}$$

$$x_t \quad \text{— Current input (e.g., a word embedding at time step } t\text{)}$$

**Outputs:**

$$C_t \quad \text{— Updated cell state (Long Term Memory)}$$

$$h_t \quad \text{— Updated hidden state (Short Term Memory)}$$

### Key Operations

| Symbol | Operation | Description |
|---|---|---|
| $\sigma$ | Sigmoid layer | A neural network layer with sigmoid activation. Outputs values in $(0, 1)$, acting as a "gate" — 0 means "block everything", 1 means "let everything through" |
| $\tanh$ | Tanh layer | A neural network layer with tanh activation. Outputs values in $(-1, 1)$, creating candidate values with both positive and negative components |
| $\odot$ | Pointwise multiplication | Element-wise (Hadamard) product of two vectors. Each element in one vector scales the corresponding element in the other |
| $+$ | Pointwise addition | Element-wise addition of two vectors. This additive operation is key to preserving gradient flow |
| $[\;,\;]$ | Concatenation | Joining two vectors end-to-end into a single longer vector |

### Vector Concatenation

The first step in every gate computation is **concatenating** the previous hidden state $h_{t-1}$ with the current input $x_t$:

$$[h_{t-1}, x_t] = \text{concat}(h_{t-1}, x_t)$$

**Example:**

$$h_{t-1} = \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} \quad \text{(3-dimensional)}$$

$$x_t = \begin{bmatrix} 2 & 3 & 4 & 5 \end{bmatrix} \quad \text{(4-dimensional)}$$

$$[h_{t-1}, x_t] = \begin{bmatrix} 1 & 2 & 3 & 2 & 3 & 4 & 5 \end{bmatrix} \quad \text{(7-dimensional)}$$

This concatenated vector serves as the input to all four gate computations. By seeing both the previous hidden state and the current input simultaneously, each gate can make decisions based on the complete available context.

### Core Components — The Four Gates

| # | Gate | Symbol | Activation | Purpose |
|---|---|---|---|---|
| 1 | **Forget Gate** | $f_t$ | Sigmoid $\sigma$ | Decides what to **remove** from long term memory |
| 2 | **Input Gate** | $i_t$ | Sigmoid $\sigma$ | Decides **how much** of the new candidate to add |
| 3 | **Candidate Memory** | $\tilde{C}_t$ | Tanh | Creates **new candidate values** to potentially store |
| 4 | **Output Gate** | $o_t$ | Sigmoid $\sigma$ | Decides what to **output** as the hidden state |

---

## 5. Working of LSTM RNN (Gates in Detail)

### 5.1 Forget Gate

> **Purpose:** Decide what information to **discard** from the cell state (long term memory).

#### Formula

$$\boxed{f_t = \sigma\left(W_f \cdot [h_{t-1}, x_t] + b_f\right)}$$

**Breaking down each component:**

| Symbol | Meaning | Shape (example) |
|---|---|---|
| $h_{t-1}$ | Hidden state of previous time step — carries short-term context from the last step | $(1 \times 3)$ |
| $x_t$ | Current input — the word/token being processed at this time step | $(1 \times 4)$ |
| $[h_{t-1}, x_t]$ | Concatenation of both — the gate sees the full available context | $(1 \times 7)$ |
| $W_f$ | Learnable weight matrix for the forget gate — these weights are trained to recognize **what contexts require forgetting** | $(7 \times 3)$ |
| $b_f$ | Bias vector for the forget gate — allows the gate to have a default tendency toward remembering or forgetting | $(1 \times 3)$ |
| $\sigma$ | Sigmoid activation function | — |
| $f_t$ | Forget gate output — a vector of values between 0 and 1 | $(1 \times 3)$ |

#### The Sigmoid Activation Function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

The sigmoid function maps any real number to the range $(0, 1)$:

- $\sigma(z) \to 0$ when $z$ is very negative → **"forget this"**
- $\sigma(z) \to 1$ when $z$ is very positive → **"remember this"**
- $\sigma(0) = 0.5$ → **"partially remember"**

This makes sigmoid the perfect activation for a **gate** — it acts like a valve that can be fully closed (0), fully open (1), or anywhere in between.

#### Dimension Walkthrough

$$\underbrace{[h_{t-1}, x_t]}_{(1 \times 7)} \cdot \underbrace{W_f}_{(7 \times 3)} + \underbrace{b_f}_{(1 \times 3)} = \underbrace{z_f}_{(1 \times 3)} \xrightarrow{\sigma} \underbrace{f_t}_{(1 \times 3)}$$

The forget gate output $f_t$ has the **same dimensionality** as the cell state $C_{t-1}$ — this is essential because they will be multiplied element-wise.

#### How the Forget Gate is Applied

The forget gate output is **element-wise multiplied** (Hadamard product) with the previous cell state:

$$f_t \odot C_{t-1}$$

Each element in $f_t$ controls how much of the corresponding element in $C_{t-1}$ survives:

#### Three Scenarios

**Scenario 1 — Complete Forgetting** (entire sentence context has changed):

$$C_{t-1} = \begin{bmatrix} 6 & 8 & 9 \end{bmatrix} \odot f_t = \begin{bmatrix} 0 & 0 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 0 & 0 \end{bmatrix}$$

> All previous context is **erased**. This happens when the gate learns that the old context is completely irrelevant (e.g., starting a new sentence with a completely different topic).

**Scenario 2 — Complete Remembering:**

$$C_{t-1} = \begin{bmatrix} 6 & 8 & 9 \end{bmatrix} \odot f_t = \begin{bmatrix} 1 & 1 & 1 \end{bmatrix} = \begin{bmatrix} 6 & 8 & 9 \end{bmatrix}$$

> All previous context is **fully preserved**. This happens when the old information is still relevant and nothing should be discarded.

**Scenario 3 — Selective Forgetting** (most common in practice):

$$C_{t-1} = \begin{bmatrix} 6 & 8 & 9 \end{bmatrix} \odot f_t = \begin{bmatrix} 0.5 & 1.0 & 0.5 \end{bmatrix} = \begin{bmatrix} 3 & 8 & 4.5 \end{bmatrix}$$

> Some dimensions are partially reduced, others fully retained. The gate has learned that **some** of the old context is still relevant while other parts should be diminished.

> **Conclusion:** Based on the current context ($h_{t-1}$ and $x_t$), the forget gate learns to selectively let go of some information or retain it. The network **learns through training** which patterns of information to forget.

---

### 5.2 Input Gate & Candidate Memory

> **Purpose:** Decide what **new information** to store in the cell state.

This step has two parts that work together:

#### Formulas

**Input Gate:**

$$\boxed{i_t = \sigma\left(W_i \cdot [h_{t-1}, x_t] + b_i\right)}$$

**Candidate Memory:**

$$\boxed{\tilde{C}_t = \tanh\left(W_C \cdot [h_{t-1}, x_t] + b_C\right)}$$

#### Input Gate — In Depth

| Symbol | Meaning | Shape |
|---|---|---|
| $W_i$ | Learnable weight matrix for the input gate — trained to recognize **what contexts require adding new info** | $(7 \times 3)$ |
| $b_i$ | Bias vector for the input gate | $(1 \times 3)$ |
| $\sigma$ | Sigmoid activation — output in $(0, 1)$ | — |
| $i_t$ | Input gate output — controls **how much** of each candidate value to actually store | $(1 \times 3)$ |

The input gate answers the question: **"How much of each new candidate value should I allow into the cell state?"**

- $i_t \to 0$: Block this candidate value — don't add it to memory
- $i_t \to 1$: Fully add this candidate value to memory

#### Candidate Memory — In Depth

| Symbol | Meaning | Shape |
|---|---|---|
| $W_C$ | Learnable weight matrix for candidate memory — trained to generate **useful new information** | $(7 \times 3)$ |
| $b_C$ | Bias vector for candidate memory | $(1 \times 3)$ |
| $\tanh$ | Tanh activation — output in $(-1, 1)$ | — |
| $\tilde{C}_t$ | Candidate values — the **proposed new information** to potentially add to the cell state | $(1 \times 3)$ |

#### Why Tanh for Candidate Memory?

The tanh function is used here instead of sigmoid for important reasons:

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

- **Range $(-1, 1)$:** Unlike sigmoid's $(0, 1)$, tanh can produce both **positive and negative** values. This means the candidate memory can both **increase and decrease** values in the cell state.
- **Zero-centered:** Tanh is centered around 0, which helps the network learn more efficiently because updates can go in either direction.
- **Stronger gradients:** The derivative of tanh ($1 - \tanh^2(z)$) has a maximum of 1, which is stronger than sigmoid's maximum derivative of 0.25.

#### How They Work Together

The input gate and candidate memory are **multiplied element-wise**:

$$i_t \odot \tilde{C}_t$$

**Example:**

$$i_t = \begin{bmatrix} 0 & 1 & 0 \end{bmatrix} \quad \tilde{C}_t = \begin{bmatrix} 2 & 8 & 5 \end{bmatrix}$$

$$i_t \odot \tilde{C}_t = \begin{bmatrix} 0 \times 2 & 1 \times 8 & 0 \times 5 \end{bmatrix} = \begin{bmatrix} 0 & 8 & 0 \end{bmatrix}$$

> The input gate **filters** the candidate values. Only the second dimension's candidate value (8) is allowed through — the other two are blocked. This is the actual new information that will be **added** to the cell state.

> **Context:** If any information needs to be added to the memory cell $C_{t-1}$, the input gate controls **how much** and the candidate memory provides **what** gets added.

---

### 5.3 Cell State Update

> **Purpose:** Combine forgetting and adding to produce the **new cell state** (updated long term memory).

#### Formula

$$\boxed{C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t}$$

This single equation is the **heart of the LSTM**. Let's break it into two parts:

**Part 1 — Forgetting (removing old information):**

$$f_t \odot C_{t-1}$$

- The forget gate $f_t$ element-wise multiplies the old cell state $C_{t-1}$
- Values close to 0 in $f_t$ erase the corresponding information
- Values close to 1 in $f_t$ preserve the corresponding information

**Part 2 — Adding (storing new information):**

$$i_t \odot \tilde{C}_t$$

- The input gate $i_t$ filters the candidate memory $\tilde{C}_t$
- Only the approved new information passes through

**The $+$ (addition) is crucial:**

The cell state is updated via **addition**, not multiplication. This is fundamentally why LSTM solves the vanishing gradient problem:

- **Multiplicative updates** (like in standard RNN): $h_t = f(W \cdot h_{t-1})$ → gradients multiply repeatedly → vanish
- **Additive updates** (LSTM cell state): $C_t = f_t \odot C_{t-1} + \text{new}$ → gradients can flow directly through the addition → preserved

#### Real-World Example

> **"I stay in India ... (many words later) ... and I speak ___"**

**Step 1 — Forget Gate decides:**
- The subject's location ("India") is relevant → **keep it** ($f_t \approx 1$ for those dimensions)
- Some earlier irrelevant details → **forget them** ($f_t \approx 0$ for those dimensions)

**Step 2 — Input Gate + Candidate adds:**
- The phrase "I speak" signals that language information is needed
- Candidate memory generates values related to language
- Input gate allows these values through

**Step 3 — Combined cell state:**
- $C_t$ now contains the preserved location context ("India") plus the new language-related context
- The network can predict: **Hindi**, **English**

> **Summary:**
>
> $$C_t = \underbrace{f_t \odot C_{t-1}}_{\text{What to keep from old memory}} + \underbrace{i_t \odot \tilde{C}_t}_{\text{What new info to add}}$$

---

### 5.4 Output Gate

> **Purpose:** Decide what information from the cell state to **output** as the hidden state (short term memory).

#### Formulas

**Output Gate:**

$$\boxed{o_t = \sigma\left(W_o \cdot [h_{t-1}, x_t] + b_o\right)}$$

**Hidden State (Short Term Memory):**

$$\boxed{h_t = o_t \odot \tanh(C_t)}$$

#### Output Gate — In Depth

| Symbol | Meaning | Shape |
|---|---|---|
| $W_o$ | Learnable weight matrix for the output gate — trained to recognize **what parts of the cell state are relevant for output** | $(7 \times 3)$ |
| $b_o$ | Bias vector for the output gate | $(1 \times 3)$ |
| $\sigma$ | Sigmoid activation — output in $(0, 1)$ | — |
| $o_t$ | Output gate value — controls **which dimensions** of the cell state to expose | $(1 \times 3)$ |

#### Hidden State Computation — In Depth

The hidden state $h_t$ is computed in two steps:

**Step 1:** The cell state $C_t$ is passed through $\tanh$:

$$\tanh(C_t)$$

This squashes the cell state values into the range $(-1, 1)$. The cell state can contain values of any magnitude (since it accumulates additions over time), so $\tanh$ normalizes these values to a manageable range.

**Step 2:** The output gate filters the normalized cell state:

$$h_t = o_t \odot \tanh(C_t)$$

This ensures that the hidden state only contains the **relevant subset** of the cell state. Not everything in long term memory needs to be output at every step.

#### Why Not Output the Cell State Directly?

The cell state $C_t$ contains **all** the accumulated long term memory. But at any given time step, only **some** of that information is relevant for the current prediction. For example:

- The cell state might remember both "the person is in India" and "it is raining"
- If the current task is predicting the next word after "I speak fluent", only the "India" part is relevant
- The output gate learns to expose only the relevant dimensions

#### Memory Flow Summary

$$C_{t-1} \xrightarrow{f_t \odot} \text{(forget)} \xrightarrow{+ \; i_t \odot \tilde{C}_t} C_t \xrightarrow{\tanh \to o_t \odot} h_t$$

| Memory Type | Carried By | Role | Lifespan |
|---|---|---|---|
| **Long Term Memory** | Cell State $C_t$ | Stores information across many time steps via additive updates | Can persist for hundreds of steps |
| **Short Term Memory** | Hidden State $h_t$ | Filtered view of cell state; used for current predictions and passed to next step | Relevant mainly for the next 1-2 steps |

> - **Forget gate** ($f_t$) → Controls what to **remove** from long term memory
> - **Input gate** ($i_t$) + **Candidate** ($\tilde{C}_t$) → Controls what to **add** to long term memory
> - **Output gate** ($o_t$) → Controls what to **expose** from long term memory as short term memory
> - **Cell state** ($C_t$) → The actual **long term memory** container
> - **Hidden state** ($h_t$) → The **short term memory** used for predictions

---

## 6. Complete LSTM Forward Pass

Here are all the LSTM equations together, executed **in order** at each time step $t$:

$$\boxed{\begin{aligned}
f_t &= \sigma\left(W_f \cdot [h_{t-1}, x_t] + b_f\right) && \text{(1) Forget Gate} \\[8pt]
i_t &= \sigma\left(W_i \cdot [h_{t-1}, x_t] + b_i\right) && \text{(2) Input Gate} \\[8pt]
\tilde{C}_t &= \tanh\left(W_C \cdot [h_{t-1}, x_t] + b_C\right) && \text{(3) Candidate Memory} \\[8pt]
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t && \text{(4) Cell State Update} \\[8pt]
o_t &= \sigma\left(W_o \cdot [h_{t-1}, x_t] + b_o\right) && \text{(5) Output Gate} \\[8pt]
h_t &= o_t \odot \tanh(C_t) && \text{(6) Hidden State}
\end{aligned}}$$

**All learnable parameters:**

| Parameter | Shape | Gate |
|---|---|---|
| $W_f, b_f$ | $(n_h + n_x) \times n_h$, $(1 \times n_h)$ | Forget Gate |
| $W_i, b_i$ | $(n_h + n_x) \times n_h$, $(1 \times n_h)$ | Input Gate |
| $W_C, b_C$ | $(n_h + n_x) \times n_h$, $(1 \times n_h)$ | Candidate Memory |
| $W_o, b_o$ | $(n_h + n_x) \times n_h$, $(1 \times n_h)$ | Output Gate |

Where $n_h$ = hidden state dimension and $n_x$ = input dimension.

**Total parameters:** $4 \times [(n_h + n_x) \times n_h + n_h] = 4 \times n_h \times (n_h + n_x + 1)$

> An LSTM has roughly **4x the parameters** of a standard RNN — one set of weights for each of the four gate/candidate computations.

---

## 7. Backpropagation & Weight Updates

### Training Objective

Given a sequence of inputs and targets, the LSTM minimizes a loss function $\mathcal{L}$ (e.g., cross-entropy for classification or MSE for regression):

$$\mathcal{L} = \sum_{t=1}^{T} \ell(\hat{y}_t, y_t)$$

Where the prediction at each step passes through a final layer:

$$\hat{y}_t = \text{softmax}(W_y \cdot h_t + b_y) \quad \text{or} \quad \hat{y}_t = \sigma(W_y \cdot h_t + b_y)$$

### Why Gradients Don't Vanish in LSTM

The key gradient to analyze is $\frac{\partial C_t}{\partial C_{t-1}}$:

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t + \frac{\partial (i_t \odot \tilde{C}_t)}{\partial C_{t-1}}$$

The dominant term is simply $f_t$. When $f_t \approx 1$ (the gate is open), the gradient is approximately **1**, meaning it passes through **without attenuation**. Over many time steps:

$$\frac{\partial C_T}{\partial C_t} = \prod_{k=t+1}^{T} f_k$$

If the forget gates are close to 1 (which the network can learn), this product stays close to 1 even over long sequences. This is in stark contrast to the standard RNN where the equivalent product involves repeated multiplication by $W_{hh}$ and $\tanh'$, which drives gradients to zero.

### Weight Update Rule

All weight matrices are updated via gradient descent:

$$W \leftarrow W - \eta \cdot \frac{\partial \mathcal{L}}{\partial W}$$

Where $\eta$ is the learning rate. The four weight matrices updated are:

| Weight | Gate | What It Learns |
|---|---|---|
| $W_f$ | Forget Gate | Patterns that signal "this old info is no longer relevant" |
| $W_i$ | Input Gate | Patterns that signal "this new info should be stored" |
| $W_C$ | Candidate Memory | How to generate useful new memory content |
| $W_o$ | Output Gate | Patterns that signal "this part of memory is relevant for the current output" |

---

## 8. GRU RNN — LSTM Variant

**GRU (Gated Recurrent Unit)**, introduced by **Cho et al. (2014)**, is a simplified variant of LSTM.

#### GRU Equations

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z) \quad \text{(Update Gate)}$$

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r) \quad \text{(Reset Gate)}$$

$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h) \quad \text{(Candidate)}$$

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad \text{(Final State)}$$

#### Key Differences from LSTM

| Feature | LSTM | GRU |
|---|---|---|
| Gates | 3 gates (forget, input, output) | 2 gates (update, reset) |
| Memory | Separate $C_t$ and $h_t$ | Single $h_t$ (merged) |
| Parameters | $4 \times n_h \times (n_h + n_x + 1)$ | $3 \times n_h \times (n_h + n_x + 1)$ |
| Forget + Input | Independent gates | Coupled via $z_t$ and $(1 - z_t)$ |
| Performance | Slightly better on very long sequences | Comparable; faster to train |

> GRU couples the forget and input mechanisms into a single **update gate** $z_t$: what you forget, you replace. This constraint reduces parameters by ~25% while often achieving comparable performance.

---

## 9. Summary

### The Problem

$$\text{Standard RNN} \xrightarrow{\text{Long Term Dependencies}} \text{Vanishing Gradient Problem}$$

$$\prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}} \approx 0 \quad \text{(gradients vanish over many steps)}$$

### The Solution — LSTM

$$\boxed{\begin{aligned}
&\text{1. Forget Gate:} & f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) & &\text{→ What to REMOVE} \\
&\text{2. Input Gate:} & i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) & &\text{→ How much to ADD} \\
&\text{3. Candidate:} & \tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) & &\text{→ What to ADD} \\
&\text{4. Cell Update:} & C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t & &\text{→ New Long Term Memory} \\
&\text{5. Output Gate:} & o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) & &\text{→ What to OUTPUT} \\
&\text{6. Hidden State:} & h_t &= o_t \odot \tanh(C_t) & &\text{→ New Short Term Memory}
\end{aligned}}$$

### Two Memory Channels

$$\underbrace{C_t}_{\text{Long Term Memory}} \quad \text{— persists across many time steps (the conveyor belt)}$$

$$\underbrace{h_t}_{\text{Short Term Memory}} \quad \text{— filtered output for immediate use and next step}$$

### Why It Works

$$\frac{\partial C_T}{\partial C_t} = \prod_{k=t+1}^{T} f_k \approx 1 \quad \text{(when forget gates are open → gradients preserved)}$$
