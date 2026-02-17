# Recurrent Neural Networks (RNN)

A **Recurrent Neural Network (RNN)** is a type of Artificial Neural Network designed to process **sequential data**. Unlike traditional feed-forward networks, RNNs have "memory"—they use information from previous inputs to influence the current input and output.

---

## 1. The Core Concept: Sequential Memory

In a standard neural network, all inputs and outputs are independent. In an RNN, the output of a step depends on the current input **and** the hidden state from the previous step.

### The Recurrent Relation

At each time step $t$, the network maintains a **hidden state** ($h_t$). This state acts as the "memory" of the network.

**The Formula:**
$$h_t = f(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = f(W_{hy} h_t + b_y)$$

**Variables:**

- $x_t$: Current input at time $t$.
- $h_{t-1}$: Hidden state from the previous time step.
- $W_{hh}$: Weight matrix for the hidden-to-hidden connection.
- $W_{xh}$: Weight matrix for the input-to-hidden connection.
- $h_t$: New hidden state (current memory).
- $f$: Activation function (usually **tanh** or **ReLU**).

---

## 2. Unrolling an RNN

To understand how an RNN processes a sequence (like the sentence "Hello world"), we "unroll" it.

- **Time Step 1:** Input "Hello" $\rightarrow$ Compute $h_1$.
- **Time Step 2:** Input "world" + $h_1$ $\rightarrow$ Compute $h_2$.

By the end of the sequence, the final hidden state $h_t$ contains a summarized representation of the entire sequence.

---

## 3. Training: Backpropagation Through Time (BPTT)

RNNs are trained using a variation of backpropagation called **Backpropagation Through Time (BPTT)**.

1. **Forward Pass:** The sequence is processed, and the loss is calculated at each time step (or at the final step).
2. **Backward Pass:** The gradient is calculated not just for the current layer, but for all previous time steps.
3. **Weight Update:** The weights ($W_{hh}, W_{xh}, W_{hy}$) are updated based on the total accumulated gradient.

---

## 4. Common RNN Architectures

| Architecture     | Example Use Case                                                |
| :--------------- | :-------------------------------------------------------------- |
| **One-to-Many**  | Image Captioning (One image $\rightarrow$ sequence of words).   |
| **Many-to-One**  | Sentiment Analysis (Sequence of words $\rightarrow$ one label). |
| **Many-to-Many** | Machine Translation (Sentence $\rightarrow$ Sentence).          |

---

## 5. The Fatal Flaws: Vanishing and Exploding Gradients

Standard RNNs struggle with **long-term dependencies** because of the way they are trained.

### The Vanishing Gradient Problem

As the error is backpropagated through many time steps, the gradient is repeatedly multiplied by the weights. If those weights are small, the gradient shrinks exponentially until it becomes zero.

- **Result:** The network "forgets" information from the beginning of a long sentence.

### The Solution: LSTM and GRU

To solve this, advanced variants were created that use "gates" to control what information to keep or discard.

- **LSTM (Long Short-Term Memory)**
- **GRU (Gated Recurrent Unit)**

---

## 6. Summary

- **Strengths:** Excellent for time-series data, speech recognition, and natural language processing.
- **Weaknesses:** Computationally slow (cannot be parallelized because steps must happen in order) and prone to vanishing gradients.
