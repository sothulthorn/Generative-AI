# Encoder-Decoder Sequence-to-Sequence (Seq2Seq) Architecture

## Table of Contents

1. [Recap: The RNN Family](#1-recap-the-rnn-family)
2. [What is Sequence-to-Sequence?](#2-what-is-sequence-to-sequence)
   - [Many-to-Many RNN Limitation](#many-to-many-rnn-limitation)
   - [The Need for Encoder-Decoder](#the-need-for-encoder-decoder)
3. [High-Level Overview](#3-high-level-overview)
   - [Simple Working](#simple-working)
   - [The Context Vector](#the-context-vector)
4. [The Encoder](#4-the-encoder)
   - [Embedding Layer](#embedding-layer)
   - [Encoder LSTM Cells](#encoder-lstm-cells)
   - [Encoding Process Step-by-Step](#encoding-process-step-by-step)
   - [What the Encoder Produces](#what-the-encoder-produces)
5. [The Decoder](#5-the-decoder)
   - [Decoder LSTM Cells](#decoder-lstm-cells)
   - [Softmax Output Layer](#softmax-output-layer)
   - [Decoding Process Step-by-Step](#decoding-process-step-by-step)
   - [Special Tokens: SOS and EOS](#special-tokens-sos-and-eos)
6. [Complete Mathematical Formulation](#6-complete-mathematical-formulation)
   - [Encoder Equations](#encoder-equations)
   - [Context Vector](#context-vector)
   - [Decoder Equations](#decoder-equations)
   - [Output Prediction](#output-prediction)
7. [Training: Teacher Forcing](#7-training-teacher-forcing)
   - [Loss Function](#loss-function)
   - [Backpropagation Through the Full Model](#backpropagation-through-the-full-model)
8. [Inference (Generation)](#8-inference-generation)
9. [Step-by-Step Walkthrough: "Thank you" to "Gracias"](#9-step-by-step-walkthrough-thank-you-to-gracias)
10. [Problems with Encoder-Decoder Seq2Seq](#10-problems-with-encoder-decoder-seq2seq)
    - [The Information Bottleneck](#the-information-bottleneck)
    - [Long Sentence Degradation](#long-sentence-degradation)
    - [BLEU Score Decay](#bleu-score-decay)
11. [The Solution: Attention Mechanism (Preview)](#11-the-solution-attention-mechanism-preview)
12. [Use Cases](#12-use-cases)
13. [Summary](#13-summary)

---

## 1. Recap: The RNN Family

The Encoder-Decoder architecture builds upon everything we have learned so far:

| # | Architecture | Key Contribution |
|---|---|---|
| 1 | **Simple RNN** | Processes sequential data with hidden state memory |
| 2 | **LSTM RNN** | Solves vanishing gradient with cell state + gates |
| 3 | **GRU RNN** | Simplified LSTM with fewer parameters |
| 4 | **Bidirectional RNN** | Captures both past and future context |
| 5 | **Encoder-Decoder (Seq2Seq)** | Maps an **input sequence** to an **output sequence** of **different length** |

The progression:

$$\text{Simple RNN} \xrightarrow{\text{Vanishing Gradient}} \text{LSTM / GRU} \xrightarrow{\text{Future Context}} \text{Bidirectional RNN} \xrightarrow{\text{Seq-to-Seq}} \text{Encoder-Decoder}$$

---

## 2. What is Sequence-to-Sequence?

A **Sequence-to-Sequence (Seq2Seq)** model takes a sequence of items as input (words, characters, tokens) and produces another sequence as output. Crucially, the input and output sequences can have **different lengths**.

$$\underbrace{(x_1, x_2, \ldots, x_m)}_{\text{Input sequence of length } m} \xrightarrow{\text{Seq2Seq}} \underbrace{(y_1, y_2, \ldots, y_n)}_{\text{Output sequence of length } n}$$

Where $m \neq n$ in general.

**Examples:**

| Task | Input Sequence | Output Sequence |
|---|---|---|
| **Language Translation** | "Thank you" (English, 2 words) | "Gracias" (Spanish, 1 word) |
| **Language Translation** | "How are you" (English, 3 words) | "Comment allez-vous" (French, 3 words) |
| **Text Summarization** | A long paragraph (100 words) | A short summary (15 words) |
| **Chatbot** | "Hi" (1 word) | "Hi, how are you?" (5 words) |
| **Text Generation** | A prompt (10 words) | A full paragraph (50 words) |

### Many-to-Many RNN Limitation

A standard Many-to-Many RNN produces **one output per input time step**:

```
  y_1    y_2    y_3
   ^      ^      ^
   |      |      |
 [h_1]--[h_2]--[h_3]
   ^      ^      ^
   |      |      |
  x_1    x_2    x_3
```

This architecture **forces** the input and output to have the **same length**. But in translation:

- "Thank you" (2 words) $\rightarrow$ "Merci beaucoup" (2 words) — works by coincidence
- "Thank you" (2 words) $\rightarrow$ "Gracias" (1 word) — **cannot work**
- "I love you" (3 words) $\rightarrow$ "Je t'aime" (2 words) — **cannot work**

### The Need for Encoder-Decoder

We need an architecture that:
1. **Reads** the entire input sequence and **compresses** it into a fixed representation
2. **Generates** the output sequence from that representation, one token at a time, stopping when it decides to

This is exactly what the Encoder-Decoder architecture does.

---

## 3. High-Level Overview

### Simple Working

The Encoder-Decoder model has two distinct components:

```
                          Context Vector
                         (Fixed-size vector)
                               |
                               v
+------------------+    [0.1, 0.8, -0.3, 0.6, 0.1]    +------------------+
|                  |         |                          |                  |
|  Input Sentence  | ------> |                   -----> |  Output Sentence |
|                  |         |                          |                  |
+------------------+    ENCODER                    DECODER  +------------------+
                   Converts words to              Generates output
                   arrays of numbers              word-by-word, feeding
                   (embeddings), processes        previous word back into
                   through LSTM cells             the decoder at each step
```

The information flows in one direction:

$$\text{Input Sentence} \xrightarrow{\text{Encoder}} \text{Context Vector} \xrightarrow{\text{Decoder}} \text{Output Sentence}$$

### The Context Vector

The **context vector** is the bridge between the encoder and decoder. It is the encoder's **final hidden state** (and cell state in LSTM), which is a fixed-size vector that represents the **entire meaning** of the input sentence.

$$\text{Context Vector} = (h_T^{\text{enc}}, C_T^{\text{enc}})$$

Where:
- $h_T^{\text{enc}}$ is the **final hidden state** (short term memory) of the encoder
- $C_T^{\text{enc}}$ is the **final cell state** (long term memory) of the encoder
- $T$ is the length of the input sequence

> The context vector is a **compressed numerical summary** of the entire input. All the meaning of "Thank you" or "How are you" must be squeezed into this single fixed-size vector. This is both the power and the limitation of this architecture.

---

## 4. The Encoder

The encoder reads the input sequence and produces the context vector. It is typically implemented as an **LSTM RNN** (though GRU or Bidirectional variants can also be used).

### Embedding Layer

Before feeding words into the LSTM, each word must be converted to a dense numerical vector. This is done by an **embedding layer**.

$$x_t = \text{Embedding}(w_t) \in \mathbb{R}^{d}$$

Where:
- $w_t$ is the input word at time step $t$ (e.g., "Thank")
- $d$ is the embedding dimension (e.g., 128, 256, 512)
- $x_t$ is the resulting dense vector

**Example with One-Hot Encoding (OHE) to Embedding:**

Suppose our vocabulary has 4 tokens: {`<SOS>`, "Thank", "you", `<EOS>`}

| Token | One-Hot Encoding | Embedding (learned, $d=4$) |
|---|---|---|
| `<SOS>` | $[1, 0, 0, 0]$ | $[0.12, -0.34, 0.56, 0.78]$ |
| Thank | $[0, 1, 0, 0]$ | $[0.45, 0.23, -0.67, 0.11]$ |
| you | $[0, 0, 1, 0]$ | $[0.89, -0.12, 0.34, -0.56]$ |
| `<EOS>` | $[0, 0, 0, 1]$ | $[0.01, 0.99, -0.01, 0.33]$ |

The embedding layer is a learnable weight matrix $W_{\text{emb}} \in \mathbb{R}^{|V| \times d}$ where $|V|$ is the vocabulary size. It transforms the sparse one-hot vector into a dense, meaningful representation where semantically similar words are closer in the embedding space.

### Encoder LSTM Cells

The encoder is an LSTM that is **unrolled** for each time step in the input. At each step, the LSTM cell:
1. Takes the current word embedding $x_t$
2. Takes the previous hidden state $h_{t-1}$ and cell state $C_{t-1}$
3. Produces updated hidden state $h_t$ and cell state $C_t$

The same LSTM layer (same weights) is applied at every time step — this is what "unrolling" means.

```
    C_0       C_1       C_2       C_3       C_4
  ----->[ LSTM ]----->[ LSTM ]----->[ LSTM ]----->[ LSTM ]----->  C_T (Long Term Memory)
    h_0  |    | h_1    |    | h_2   |    | h_3    |    | h_4
  ----->[ LSTM ]----->[ LSTM ]----->[ LSTM ]----->[ LSTM ]----->  h_T (Short Term Memory)
          ^             ^            ^             ^
          |             |            |             |
         x_1           x_2         x_3           x_4
        <SOS>         Thank        you           <EOS>
         t=1           t=2         t=3            t=4
```

### Encoding Process Step-by-Step

For the input sentence **"Thank you"** (with special tokens: `<SOS>` Thank you `<EOS>`):

| Time Step | Input Token | Input Embedding | Operation | Output |
|---|---|---|---|---|
| $t=1$ | `<SOS>` | $x_1 = \text{Embed}(\text{SOS})$ | LSTM$(x_1, h_0, C_0)$ | $h_1, C_1$ |
| $t=2$ | Thank | $x_2 = \text{Embed}(\text{Thank})$ | LSTM$(x_2, h_1, C_1)$ | $h_2, C_2$ |
| $t=3$ | you | $x_3 = \text{Embed}(\text{you})$ | LSTM$(x_3, h_2, C_2)$ | $h_3, C_3$ |
| $t=4$ | `<EOS>` | $x_4 = \text{Embed}(\text{EOS})$ | LSTM$(x_4, h_3, C_3)$ | $h_4, C_4$ |

Where $h_0 = \mathbf{0}$ and $C_0 = \mathbf{0}$ (zero initialization).

### What the Encoder Produces

After processing the entire input sequence, the encoder's outputs are:

| Output | Symbol | Role |
|---|---|---|
| **Final Hidden State** | $h_T^{\text{enc}}$ | Short term memory — carries the context of this sentence |
| **Final Cell State** | $C_T^{\text{enc}}$ | Long term memory — carries deeper, long-range information |

Together, these form the **context vector**:

$$\boxed{\text{Context Vector} = (h_T^{\text{enc}}, \; C_T^{\text{enc}})}$$

> **Critical point:** All the intermediate hidden states $(h_1, h_2, \ldots, h_{T-1})$ are **discarded**. Only the **final states** are passed to the decoder. The entire input sentence must be represented in this single pair of vectors.

---

## 5. The Decoder

The decoder takes the context vector and generates the output sequence **one token at a time**, in an autoregressive manner (each output is fed back as input for the next step).

### Decoder LSTM Cells

The decoder is a **separate LSTM** (with its own learnable weights) that is initialized with the encoder's context vector:

$$h_0^{\text{dec}} = h_T^{\text{enc}} \qquad C_0^{\text{dec}} = C_T^{\text{enc}}$$

This is how the encoder **transfers knowledge** to the decoder — the decoder starts with the encoder's "understanding" of the input sentence.

### Softmax Output Layer

At each time step, the decoder's hidden state is passed through a **fully connected layer** followed by a **softmax activation** to produce a probability distribution over the entire target vocabulary:

$$P(y_t | y_{<t}, \mathbf{x}) = \text{softmax}(W_o \cdot h_t^{\text{dec}} + b_o)$$

Where:
- $W_o \in \mathbb{R}^{n_h \times |V_{\text{target}}|}$ is the output weight matrix
- $|V_{\text{target}}|$ is the target vocabulary size
- The output is a probability for every word in the vocabulary
- The word with the highest probability is selected as the prediction

**Example:** If the target vocabulary is {`<SOS>`, `<EOS>`, "Perro", "Gracias", "Gato"}, the softmax might output:

$$\hat{y}_1 = \text{softmax}(W_o \cdot h_1^{\text{dec}}) = [0.01, 0.02, 0.05, \mathbf{0.90}, 0.02]$$

The model predicts **"Gracias"** (index 3) with 90% confidence.

### Decoding Process Step-by-Step

For translating **"Thank you"** $\rightarrow$ **"Gracias"**:

```
                      Context Vector
                    (h_T^enc, C_T^enc)
                          |
                          v
  [Softmax]          [Softmax]          [Softmax]
      ^                  ^                  ^
      |                  |                  |
  [ LSTM ]----------[ LSTM ]----------[ LSTM ]
      ^                  ^                  ^
      |                  |                  |
    <SOS>            "Gracias"           <EOS>
     t=1               t=2               t=3

  Output:            Output:            Output:
  "Gracias"          <EOS>              (stop)
```

| Time Step | Input Token | Hidden State Init | Softmax Output | Predicted Token |
|---|---|---|---|---|
| $t=1$ | `<SOS>` | $(h_T^{\text{enc}}, C_T^{\text{enc}})$ from encoder | $[0.01, 0.02, 0.05, \mathbf{0.90}, 0.02]$ | **Gracias** |
| $t=2$ | Gracias | $(h_1^{\text{dec}}, C_1^{\text{dec}})$ | $[0.01, \mathbf{0.95}, 0.01, 0.02, 0.01]$ | **\<EOS\>** |
| $t=3$ | — | — | — | **STOP** (EOS was generated) |

### Special Tokens: SOS and EOS

| Token | Full Name | Purpose |
|---|---|---|
| **`<SOS>`** (or `<BOS>`) | Start of Sequence | Signals the decoder to **begin generating**. Always the first input to the decoder. |
| **`<EOS>`** | End of Sequence | Signals the decoder to **stop generating**. When predicted, the output sequence is complete. Also marks the end of the encoder input. |

**Why `<SOS>` is needed:** The decoder generates tokens autoregressively — each step's input is the previous step's output. But at $t=1$, there is no previous output. The `<SOS>` token serves as the "seed" input to kick off generation.

**Why `<EOS>` is needed:** The decoder doesn't know how long the output should be. It could be 1 word or 100 words. The `<EOS>` token is a learnable signal that the model predicts when it has finished generating the complete output.

---

## 6. Complete Mathematical Formulation

### Encoder Equations

Given input sequence $(x_1, x_2, \ldots, x_T)$ after embedding:

$$\boxed{\begin{aligned}
f_t^{\text{enc}} &= \sigma(W_f^{\text{enc}} \cdot [h_{t-1}^{\text{enc}}, x_t] + b_f^{\text{enc}}) && \text{(Forget Gate)} \\[6pt]
i_t^{\text{enc}} &= \sigma(W_i^{\text{enc}} \cdot [h_{t-1}^{\text{enc}}, x_t] + b_i^{\text{enc}}) && \text{(Input Gate)} \\[6pt]
\tilde{C}_t^{\text{enc}} &= \tanh(W_C^{\text{enc}} \cdot [h_{t-1}^{\text{enc}}, x_t] + b_C^{\text{enc}}) && \text{(Candidate Memory)} \\[6pt]
C_t^{\text{enc}} &= f_t^{\text{enc}} \odot C_{t-1}^{\text{enc}} + i_t^{\text{enc}} \odot \tilde{C}_t^{\text{enc}} && \text{(Cell State Update)} \\[6pt]
o_t^{\text{enc}} &= \sigma(W_o^{\text{enc}} \cdot [h_{t-1}^{\text{enc}}, x_t] + b_o^{\text{enc}}) && \text{(Output Gate)} \\[6pt]
h_t^{\text{enc}} &= o_t^{\text{enc}} \odot \tanh(C_t^{\text{enc}}) && \text{(Hidden State)}
\end{aligned}}$$

Applied for $t = 1, 2, \ldots, T$ with $h_0^{\text{enc}} = \mathbf{0}$ and $C_0^{\text{enc}} = \mathbf{0}$.

### Context Vector

$$\boxed{\text{Context Vector:} \quad h_0^{\text{dec}} = h_T^{\text{enc}}, \quad C_0^{\text{dec}} = C_T^{\text{enc}}}$$

The encoder's final states are directly assigned as the decoder's initial states. This is the **only connection** between the encoder and decoder in the basic Seq2Seq architecture.

### Decoder Equations

Given the previous decoder output $y_{t-1}$ (embedded) and previous states:

$$\boxed{\begin{aligned}
f_t^{\text{dec}} &= \sigma(W_f^{\text{dec}} \cdot [h_{t-1}^{\text{dec}}, y_{t-1}] + b_f^{\text{dec}}) && \text{(Forget Gate)} \\[6pt]
i_t^{\text{dec}} &= \sigma(W_i^{\text{dec}} \cdot [h_{t-1}^{\text{dec}}, y_{t-1}] + b_i^{\text{dec}}) && \text{(Input Gate)} \\[6pt]
\tilde{C}_t^{\text{dec}} &= \tanh(W_C^{\text{dec}} \cdot [h_{t-1}^{\text{dec}}, y_{t-1}] + b_C^{\text{dec}}) && \text{(Candidate Memory)} \\[6pt]
C_t^{\text{dec}} &= f_t^{\text{dec}} \odot C_{t-1}^{\text{dec}} + i_t^{\text{dec}} \odot \tilde{C}_t^{\text{dec}} && \text{(Cell State Update)} \\[6pt]
o_t^{\text{dec}} &= \sigma(W_o^{\text{dec}} \cdot [h_{t-1}^{\text{dec}}, y_{t-1}] + b_o^{\text{dec}}) && \text{(Output Gate)} \\[6pt]
h_t^{\text{dec}} &= o_t^{\text{dec}} \odot \tanh(C_t^{\text{dec}}) && \text{(Hidden State)}
\end{aligned}}$$

Where $y_0 = \text{Embed}(\text{SOS})$.

> Note: The encoder and decoder have **separate weight matrices**. $W_f^{\text{enc}} \neq W_f^{\text{dec}}$, etc. They are two independent LSTMs that only communicate through the context vector.

### Output Prediction

At each decoder time step:

$$\boxed{\hat{y}_t = \text{softmax}(W_o \cdot h_t^{\text{dec}} + b_o)}$$

$$\boxed{\text{Predicted token:} \quad w_t = \arg\max(\hat{y}_t)}$$

---

## 7. Training: Teacher Forcing

During training, we know the correct target sequence (ground truth). **Teacher forcing** is a training technique where the decoder receives the **actual correct previous token** as input, rather than its own (potentially wrong) prediction.

**Without teacher forcing** (autoregressive, used at inference):

$$\text{Input at } t: \quad \hat{y}_{t-1} \quad \text{(model's own prediction — may be wrong)}$$

**With teacher forcing** (used during training):

$$\text{Input at } t: \quad y_{t-1}^{\text{true}} \quad \text{(ground truth — always correct)}$$

**Example:** Translating "Thank you" $\rightarrow$ "Gracias"

| Step | Without Teacher Forcing | With Teacher Forcing |
|---|---|---|
| $t=1$ | Input: `<SOS>`, Predict: "Gracias" | Input: `<SOS>`, Predict: "Gracias" |
| $t=2$ | Input: "Gracias" (correct!), Predict: `<EOS>` | Input: "Gracias" (ground truth), Predict: `<EOS>` |

If the model predicted wrongly at $t=1$ (say "Perro" instead of "Gracias"):

| Step | Without Teacher Forcing | With Teacher Forcing |
|---|---|---|
| $t=1$ | Input: `<SOS>`, Predict: "Perro" (wrong!) | Input: `<SOS>`, Predict: "Perro" (wrong!) |
| $t=2$ | Input: **"Perro"** (wrong input propagates!) | Input: **"Gracias"** (ground truth rescues!) |

> Teacher forcing prevents **error accumulation** during training. Without it, one wrong prediction cascades through all subsequent time steps, making learning very difficult. With it, the decoder always gets the correct context, enabling faster and more stable training.

### Loss Function

The loss is computed as the **cross-entropy** between the predicted probability distributions and the true target tokens, summed across all decoder time steps:

$$\boxed{\mathcal{L} = -\sum_{t=1}^{n} \sum_{v=1}^{|V|} y_{t,v}^{\text{true}} \cdot \log(\hat{y}_{t,v})}$$

Where:
- $n$ is the length of the target sequence
- $|V|$ is the target vocabulary size
- $y_{t,v}^{\text{true}}$ is 1 if the true token at step $t$ is word $v$, else 0 (one-hot)
- $\hat{y}_{t,v}$ is the predicted probability of word $v$ at step $t$

Since $y_{t,v}^{\text{true}}$ is one-hot, this simplifies to:

$$\mathcal{L} = -\sum_{t=1}^{n} \log(\hat{y}_{t, \; w_t^{\text{true}}})$$

Where $w_t^{\text{true}}$ is the index of the correct word at step $t$. This means: maximize the probability assigned to the correct word at every step.

### Backpropagation Through the Full Model

Gradients flow **backward through the entire model**:

$$\mathcal{L} \xrightarrow{\nabla} \text{Decoder Softmax} \xrightarrow{\nabla} \text{Decoder LSTM} \xrightarrow{\nabla} \text{Context Vector} \xrightarrow{\nabla} \text{Encoder LSTM} \xrightarrow{\nabla} \text{Embedding Layer}$$

All parameters are updated jointly via an optimizer (Adam, SGD, etc.):

$$\theta \leftarrow \theta - \eta \cdot \frac{\partial \mathcal{L}}{\partial \theta}$$

Where $\theta$ includes:
- Encoder LSTM weights: $W_f^{\text{enc}}, W_i^{\text{enc}}, W_C^{\text{enc}}, W_o^{\text{enc}}$ and biases
- Decoder LSTM weights: $W_f^{\text{dec}}, W_i^{\text{dec}}, W_C^{\text{dec}}, W_o^{\text{dec}}$ and biases
- Output layer: $W_o, b_o$
- Embedding matrices for both source and target languages

---

## 8. Inference (Generation)

At inference time (when we don't have the ground truth), the decoder operates **autoregressively**:

1. Feed `<SOS>` as the first input
2. Get the predicted token $\hat{y}_1$
3. Feed $\hat{y}_1$ as the next input
4. Get the predicted token $\hat{y}_2$
5. Repeat until `<EOS>` is predicted or a maximum length is reached

```
                Context Vector
                     |
                     v
  <SOS> --> [LSTM] --> softmax --> "Gracias"
                                      |
            "Gracias" --> [LSTM] --> softmax --> <EOS>
                                                  |
                                                STOP
```

> Unlike training (where teacher forcing provides the correct input at every step), inference must rely entirely on the model's own predictions. This is why training quality matters so much — errors compound during autoregressive generation.

---

## 9. Step-by-Step Walkthrough: "Thank you" to "Gracias"

Let's trace the complete flow of translating **English "Thank you"** to **Spanish "Gracias"**.

### Dataset Setup

| Source (English) | Target (Spanish) |
|---|---|
| `<SOS>` Thank you `<EOS>` | `<SOS>` Gracias `<EOS>` |

### Phase 1 — Encoder Processes Input

**Step 1:** Embed each input token

$$x_1 = \text{Embed}(\text{SOS}), \quad x_2 = \text{Embed}(\text{Thank}), \quad x_3 = \text{Embed}(\text{you}), \quad x_4 = \text{Embed}(\text{EOS})$$

**Step 2:** Process through encoder LSTM

| $t$ | Token | LSTM Computation | Result |
|---|---|---|---|
| 1 | `<SOS>` | $\text{LSTM}(x_1, h_0, C_0)$ | $h_1^{\text{enc}}, C_1^{\text{enc}}$ — knows "start" |
| 2 | Thank | $\text{LSTM}(x_2, h_1, C_1)$ | $h_2^{\text{enc}}, C_2^{\text{enc}}$ — knows "Thank" |
| 3 | you | $\text{LSTM}(x_3, h_2, C_2)$ | $h_3^{\text{enc}}, C_3^{\text{enc}}$ — knows "Thank you" |
| 4 | `<EOS>` | $\text{LSTM}(x_4, h_3, C_3)$ | $h_4^{\text{enc}}, C_4^{\text{enc}}$ — **context of entire sentence** |

**Step 3:** Extract context vector

$$\text{Context Vector} = (h_4^{\text{enc}}, \; C_4^{\text{enc}})$$

> At this point, the meaning of "Thank you" is compressed into two fixed-size vectors.

### Phase 2 — Decoder Generates Output

**Step 4:** Initialize decoder with context vector

$$h_0^{\text{dec}} = h_4^{\text{enc}}, \quad C_0^{\text{dec}} = C_4^{\text{enc}}$$

**Step 5:** Decode token by token

| $t$ | Input Token | LSTM Computation | Softmax Output | Prediction |
|---|---|---|---|---|
| 1 | `<SOS>` | $\text{LSTM}(\text{Embed(SOS)}, h_0^{\text{dec}}, C_0^{\text{dec}})$ | $[0.01, 0.02, 0.05, \mathbf{0.90}, 0.02]$ | **Gracias** |
| 2 | Gracias | $\text{LSTM}(\text{Embed(Gracias)}, h_1^{\text{dec}}, C_1^{\text{dec}})$ | $[0.01, \mathbf{0.95}, 0.01, 0.02, 0.01]$ | **\<EOS\>** |

**Step 6:** `<EOS>` predicted $\rightarrow$ **STOP**

$$\text{Output: "Gracias"} \quad \checkmark$$

### Phase 3 — Training (Loss & Update)

**Step 7:** Compute loss

$$\mathcal{L} = -\log P(\text{Gracias} | \text{SOS}, \text{context}) - \log P(\text{EOS} | \text{Gracias}, \text{context})$$

$$\mathcal{L} = -\log(0.90) - \log(0.95) = 0.105 + 0.051 = 0.156$$

**Step 8:** Backpropagate and update all weights

$$\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}$$

---

## 10. Problems with Encoder-Decoder Seq2Seq

### The Information Bottleneck

The fundamental problem is that the **entire input sentence** must be compressed into a **single fixed-size context vector**:

$$\underbrace{(x_1, x_2, \ldots, x_T)}_{\text{Entire input (could be 100 words)}} \xrightarrow{\text{compress}} \underbrace{(h_T^{\text{enc}}, C_T^{\text{enc}})}_{\text{Fixed-size vector (e.g., 256 or 512 dimensions)}}$$

For short sentences (5-10 words), this works well. But for long sentences:

- A 256-dimensional vector cannot faithfully represent a 100-word paragraph
- Information from early time steps gets diluted or lost
- The decoder has no way to "look back" at specific parts of the input

> Think of it like asking someone to read an entire book, then summarize it in a single sentence, and then asking someone else to reconstruct the book from that sentence alone. Information loss is inevitable.

### Long Sentence Degradation

The encoder processes tokens sequentially. Due to the recurrent nature, information from earlier tokens gets progressively overwritten:

| Input Position | Distance from Context Vector | Information Preserved |
|---|---|---|
| Last few tokens | Close | High |
| Middle tokens | Medium | Moderate |
| First few tokens | Far | Low (diluted) |

Even with LSTM's cell state protecting long-range information, there is a practical limit. Research papers (Sutskever et al., 2014; Cho et al., 2014) showed that performance degrades significantly for sentences longer than about 30-40 tokens.

### BLEU Score Decay

The **BLEU score** (Bilingual Evaluation Understudy) is the standard metric for evaluating translation quality. Research showed a clear pattern:

```
  BLEU
  Score
    ^
    |
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

- **Short sentences (10-20 words):** High BLEU scores — the context vector can represent them well
- **Medium sentences (20-30 words):** Peak performance — good balance between context and length
- **Long sentences (40+ words):** Sharp decline — the context vector becomes a bottleneck

> This BLEU score decay for longer sentences is the primary evidence that the fixed-size context vector is insufficient. It directly motivated the development of the **Attention Mechanism**.

---

## 11. The Solution: Attention Mechanism (Preview)

The Attention Mechanism (Bahdanau et al., 2014) addresses the bottleneck by allowing the decoder to **look at all encoder hidden states**, not just the last one.

**Basic Seq2Seq:**

$$\text{Decoder sees:} \quad h_T^{\text{enc}} \text{ only (one vector)}$$

**Seq2Seq with Attention:**

$$\text{Decoder sees:} \quad (h_1^{\text{enc}}, h_2^{\text{enc}}, \ldots, h_T^{\text{enc}}) \text{ (all vectors, weighted by relevance)}$$

Instead of a single context vector, the attention mechanism computes a **different weighted combination** of all encoder states for **each decoder step**:

$$c_t = \sum_{i=1}^{T} \alpha_{t,i} \cdot h_i^{\text{enc}}$$

Where $\alpha_{t,i}$ are learned attention weights that indicate how much the decoder at step $t$ should "attend to" encoder position $i$.

> For longer paragraphs, Attention provides both a **context vector** (global summary) and **direct access to specific positions** (local details) — solving the information bottleneck.

---

## 12. Use Cases

| Application | Input | Output | Example |
|---|---|---|---|
| **Language Translation** | Sentence in language A | Sentence in language B | "Thank you" $\rightarrow$ "Gracias" |
| **Text Summarization** | Long document | Short summary | News article $\rightarrow$ Headline |
| **Chatbot / Dialogue** | User message | Bot response | "Hi" $\rightarrow$ "Hi, how are you?" |
| **Text Generation** | Prompt | Continuation | "Once upon a" $\rightarrow$ "time there was a..." |
| **Speech Recognition** | Audio features (sequence) | Text (sequence) | Audio waveform $\rightarrow$ "Hello world" |
| **Image Captioning** | Image features (CNN) | Caption (sequence) | Image of a cat $\rightarrow$ "A cat eating food" |
| **Code Generation** | Natural language description | Code | "Sort a list" $\rightarrow$ `list.sort()` |
| **Text Suggestion** | Partial text | Completed text | "Dear Sir," $\rightarrow$ "I am writing to..." |

---

## 13. Summary

### The Architecture

$$\boxed{\text{Input Sequence} \xrightarrow{\text{Embedding}} \xrightarrow{\text{Encoder LSTM}} \text{Context Vector} \xrightarrow{\text{Decoder LSTM}} \xrightarrow{\text{Softmax}} \text{Output Sequence}}$$

### Two Components

| Component | Role | Input | Output |
|---|---|---|---|
| **Encoder** | Reads and compresses the input sequence | $(x_1, x_2, \ldots, x_T)$ | Context vector $(h_T^{\text{enc}}, C_T^{\text{enc}})$ |
| **Decoder** | Generates the output sequence token-by-token | Context vector + `<SOS>` | $(y_1, y_2, \ldots, y_n,$ `<EOS>`$)$ |

### Key Equations

$$\boxed{\begin{aligned}
&\text{Encoder:} & h_t^{\text{enc}}, C_t^{\text{enc}} &= \text{LSTM}_{\text{enc}}(x_t, h_{t-1}^{\text{enc}}, C_{t-1}^{\text{enc}}) && \text{for } t = 1 \ldots T \\[8pt]
&\text{Bridge:} & h_0^{\text{dec}} &= h_T^{\text{enc}}, \quad C_0^{\text{dec}} = C_T^{\text{enc}} && \text{(context vector transfer)} \\[8pt]
&\text{Decoder:} & h_t^{\text{dec}}, C_t^{\text{dec}} &= \text{LSTM}_{\text{dec}}(y_{t-1}, h_{t-1}^{\text{dec}}, C_{t-1}^{\text{dec}}) && \text{for } t = 1 \ldots n \\[8pt]
&\text{Output:} & \hat{y}_t &= \text{softmax}(W_o \cdot h_t^{\text{dec}} + b_o) && \text{(probability over vocab)}
\end{aligned}}$$

### The Bottleneck Problem

$$\underbrace{\text{Entire input sentence}}_{\text{variable length}} \xrightarrow{\text{compress}} \underbrace{\text{Fixed-size context vector}}_{\text{information loss for long sequences}} \xrightarrow{\text{generate}} \underbrace{\text{Output sentence}}_{\text{degrades with input length}}$$

### The Evolution

$$\text{Seq2Seq} \xrightarrow{\text{bottleneck problem}} \text{Seq2Seq + Attention} \xrightarrow{\text{remove recurrence entirely}} \text{Transformer}$$
