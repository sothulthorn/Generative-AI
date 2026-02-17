# Artificial Neural Networks (ANN)

An **Artificial Neural Network (ANN)** is a computational model inspired by the biological neural networks in the human brain. It is the foundation of Deep Learning.

---

## 1. The Building Block: The Artificial Neuron

A single neuron (often called a **Perceptron**) is the basic unit of an ANN. It performs three main operations:

1. **Weighted Sum**: Inputs ($x_i$) are multiplied by weights ($w_i$).
2. **Bias ($b$)**: A constant is added to the sum to provide flexibility.
3. **Activation Function ($f$)**: A non-linear function is applied to the result.

### Mathematical Formula:

$$z = \sum_{i=1}^{n} (w_i \cdot x_i) + b$$
$$\text{Output} = f(z)$$

---

## 2. Network Architecture

ANNs are organized into three distinct types of layers:

| Layer Type          | Purpose                                                        |
| :------------------ | :------------------------------------------------------------- |
| **Input Layer**     | Receives the raw data (features). No computation happens here. |
| **Hidden Layer(s)** | Performs intermediate computations and extracts patterns.      |
| **Output Layer**    | Produces the final prediction (e.g., probability or value).    |

---

## 3. The Learning Process

The network learns by adjusting its weights and biases through an iterative cycle:

### Step 1: Forward Propagation

Data flows from input to output. The network generates a prediction based on current weights.

### Step 2: Loss Function

The error is calculated by comparing the prediction ($\hat{y}$) to the actual target ($y$). Common functions include:

- **Mean Squared Error (MSE)** for regression.
- **Cross-Entropy Loss** for classification.

### Step 3: Backpropagation

The network calculates the **gradient** of the loss function with respect to each weight using the **Chain Rule**. It determines how much each weight contributed to the error.

### Step 4: Weight Update (Optimization)

An optimizer (like SGD or Adam) updates the weights to minimize the loss:
$$w_{new} = w_{old} - \eta \cdot \frac{\partial \text{Loss}}{\partial w}$$
_(Where $\eta$ is the **Learning Rate**)_

---

## 4. Activation Functions

Activation functions allow the network to learn complex, non-linear relationships.

- **ReLU (Rectified Linear Unit)**: $f(z) = \max(0, z)$ — Most common for hidden layers.
- **Sigmoid**: $f(z) = \frac{1}{1 + e^{-z}}$ — Used for binary classification.
- **Softmax**: Normalizes outputs into probabilities that sum to 1.

---

## 5. Key Terminology

- **Epoch**: One complete pass of the training dataset through the network.
- **Learning Rate**: A hyperparameter that controls the step size during optimization.
- **Vanishing Gradient**: A problem in deep networks where gradients become so small that the weights stop updating.
