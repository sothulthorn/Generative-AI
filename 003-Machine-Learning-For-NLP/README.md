# Machine Learning for NLP

## Table of Contents

1. [One Hot Encoding](#one-hot-encoding)
2. [Bag of Words](#bag-of-words)
3. [N-Grams](#n-grams)
4. [TF-IDF](#tf-idf)
5. [Word Embedding](#word-embedding)
   - [Word2Vec](#word2vec)
     - [Continuous Bag of Words (CBOW)](#continuous-bag-of-words-cbow)
     - [Skip-Gram](#skip-gram)
     - [Average Word2Vec](#average-word2vec)

## One Hot Encoding

**One-Hot Encoding** is a way to convert **categorical data** (text labels like colors, types) into a numeric format that machine learning models can understand.

Instead of assigning categories random numbers (which can create fake "order"), one-hot encoding create a separate binary column for each category:

- 1 = category is present
- 0 = category is not present

### Why we use it

Most ML models only work with numbers.
But categories like `"Red"`, `"Blue"`, `"Green"` are not naturally numeric.

If you encode them as:

- Red = 1, Blue = 2, Green = 3 → That wrongly suggests **Green > Blue > Red**.

One-hot encoding avoid this problem.

### Example

**Original Data**
| Person | Color |
| ------ | ----- |
| A | Red |
| B | Blue |
| C | Green |
| D | Blue |

**One-Hot Encoded Output**
| Person | Color_Red | Color_Blue | Color_Green |
| ------ | --------- | ---------- | ----------- |
| A | 1 | 0 | 0 |
| B | 0 | 1 | 0 |
| C | 0 | 0 | 1 |
| D | 0 | 1 | 0 |

### Advantage and Disadvantage

| Advantage                     | Disadvantage                             |
| ----------------------------- | ---------------------------------------- |
| Easy to implement with python | Sparse metrics → Overfitting             |
|                               | ML Algorithm → Fixed size input          |
|                               | No semantics meaning is getting captured |
|                               | Out of vocabulary (OOV)                  |

## Bag of Words

**Bag of Words (BoW)** is a simple NLP technique that converts text into numbers by counting how many times each word appears.

It ignores:

- grammar
- word order
- sentence structure

It only keeps:

- word frequency

### Why it's called "Bag of Words"

Because the sentence is treated like a bag of words:

- words are throw in
- order doesn't matter
- only counts matter

### Example

Sentences

1. `"I love NLP"`
2. `"I love AI"`

**Step 1: Build Vocabulary (unique word)**

Vocabulary = `["I", "love", "NLP", "AI"]`

**Step 2: Convert each sentence into a vector (word count)**

| Sentence   | I   | love | NLP | AI  |
| ---------- | --- | ---- | --- | --- |
| I love NLP | 1   | 1    | 1   | 0   |
| I love AI  | 1   | 1    | 0   | 1   |

So the vectors become:

- `"I love NLP"` → [1, 1, 1, 0]
- `"I love AI"` → [1, 1, 0, 1]

### Advantage and Disadvantage

| Advantage                        | Disadvantage                            |
| -------------------------------- | --------------------------------------- |
| Simple and Intuitive             | Sparse metrics or array → Overfitting   |
| Fixed sized input → ML Algorithm | Ordering of the word is getting changed |
|                                  | Semantic meaning is still not captured  |
|                                  | Out of vocabulary (OOV)                 |

## N-Grams

**N-Grams** are sequences of **N consecutive words (or characters)** from a text.

They are used to capture **some context** (word order), unlike Bag of Words.

**Types of N-Grams**

- **Unigram (1-gram)** → single words
  - Example: `["I", "love", "NLP"]`
- **Bigram (2-gram)** → pairs of consecutive words
  - Example: `["I love", "love NLP"]`
- **Trigram (3-gram)** → triples of consecutive works
  - Example: `["I love NLP"]`

### Example

Text:

**"I love natural language processing"**

**Unigrams (N=1**)

`["I", "love", "natural", "language", "processing"]`

**Bigrams (N=2**)

`["I love", "love natural", "natural language", "language processing"]`

**Trigrams (N=3**)

`["I love natural", "love natural language", "natural language processing"]`

### Why N-Grams are useful\*\*

They help models understand **local word order** and **phrases**.

Examples:

- `not good` has different meaning than `good` → BoW may miss, but bigrams capture it.

## TF-IDF

**TF-IDF (Term Frequency-Inverse Document Frequency)** is a text vectorization method that converts text into numbers by giving each word a score based on:

- **TF** : how often the word appears in a document
- **IDF** : how rare the word is across all documents

So TF-IDF gives **high scores to important words,** and **low scores to common words** like _"the"_, _"is"_, _"and"_.

### 1. Term Frequency (TF)

Measures how frequently a word appears in a document.

$$TF(t,d) = \frac{\text{count of term } t \text{ in document } d}{\text{total number of terms in document } d}$$

### 2. Inverse Document Frequency (IDF)

Measures how rare a word is across documents.

$$IDF(t) = \log\left(\frac{N}{df(t)}\right)$$

Where:

- $N$ = total number of documents
- $df(t)$ = number of documents containing the term $t$

### TF-IDF Formula

$$TF\text{-}IDF(t,d) = TF(t,d) \times IDF(t)$$

### Example

**Documents**

1. D1: `"I love NLP"`
2. D2: `"I love AI"`

Vocabulary = `["I", "love", "NLP", "AI"]`

**Step 1: TF (counts)**
| Word | TF in D1 | TF in D2 |
| ---- | -------- | -------- |
| I | 1 | 1 |
| love | 1 | 1 |
| NLP | 1 | 0 |
| AI | 0 | 1 |

**Step 2: Document Frequency (df)**
| Word | df |
| ---- | -- |
| I | 2 |
| love | 2 |
| NLP | 1 |
| AI | 1 |

**Step 3: IDF Intuition**

- Words in all **documents** (`I`, `love`) → low IDF
- Words in **only one document** (`NLP`, `AI`) → high IDF

So TF-IDF will highlight:

- `NLP` in D1
- `AI` in D2

### Advantage and Disadvantage

| Advantage                           | Disadvantage            |
| ----------------------------------- | ----------------------- |
| Intuitive                           | Sparsing still exists   |
| Fixed sized → Vocabulary size       | Ouf of vocabulary (OOV) |
| Word importance is getting captured |                         |

## Word Embedding

In natural language processing (NLP), word embedding is a term used for the representation of words for text analysis, typically in the form a real-valued vector that encodes that meaning of the word such that the words that are closer in the vector space are expected to be similar in meaning.

### Why Word Embeddings are used

Because they help models understand relationships like:

- king $\approx$ queen
- Paris $\approx$ France
- good $\approx$ nice

This is something BoW and TF-IDF cannot do.

## Word2Vec

Word2Vec is a technique for natural language processing published in 2013. The word2vec algorightm uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence. As the name implies, word2vec represents each distinct word with a particular list of numbers called a vector.

### Why Word2Vec is important

Unlike BoW / TF-IDF, Word2Vec:

- captures **semantic meaning**
- produces **dense vectors** (ex: 100–300 dimensions)
- learns word relationships like:
  - **king − man + woman ≈ queen**

### Example (Conceptual)

Suppose we embed words into 3D vectors (real models use 50–300 dims):

- `cat → [0.21, 0.77, 0.10]`
- `dog → [0.25, 0.74, 0.08]`
- `car → [0.90, 0.10, 0.30]`

Here, cat and dog vectors are close → meaning is similar.
car is far away → meaning is different.

### Key Idea

Word embeddings capture meaning using **distance**.

- close vectors = similar meaning
- far vectors = different meaning

Usually measured using **cosine similarity**

### Advantage and Disadvantage

- Sparse matrix → Dense matrix
- Semantic info is getting captured
- Vocabulary size → Fixed set of dimension (Google word2vec - 300 dimension)
- Out of vocabulary (OOV) is also solved

## Continuous Bag of Words (CBOW)

CBOW tries to predict the **center word** from the **surrounding context words**.

### Example

Sentence:

> “I love natural language processing”

Take `"natural"` as the target words.

With window size = 2:

Context words:

- `"I"`, `"love"`, `"language"`, `"processing"`

Target:

- `"natural"`

So training example looks like:

Input: `["I", "love", "language", "processing"]`

Output: `"natural"`

### How CBOW works internally

#### Notation

- Vocabulary size: $V$
- Embedding dimension: $D$
- Context window size (number of context words): $C$

#### Text

Sentence:

> I love NLP

We'll predict the center word **"love"** from context words:

Context = `["I", "NLP"]`

Target = `"love"`

#### Step 0: Define a tiny vocabulary

Let vocabulary be:
| Word | Index |
| ---- | ----- |
| I | 0 |
| love | 1 |
| NLP | 2 |
| is | 3 |
| fun | 4 |

So:
$$V = 5$$

Choose embedding size:
$$D = 2$$

#### Step 1: Define embedding matrix $W$

$$W = \begin{bmatrix} 0.1 & 0.3 \\ 0.0 & 0.2 \\ 0.4 & 0.1 \\ 0.2 & 0.0 \\ 0.3 & 0.5 \end{bmatrix}$$

Each row corresponds to a word embedding:

- $v_{I} = [0.1, 0.3]$
- $v_{NLP} = [0.4, 0.1]$

#### Step 2: Get Embeddings for Context Words

Context Words: **I, NLP**

$$v_{I} = [0.1, 0.3]$$
$$v_{NLP} = [0.4, 0.1]$$

#### Step 3: Average them to get hidden vector $h$

To find the hidden layer representation, we take the average of the context word embeddings.

**Formula:**
$$h = \frac{1}{2}(v_{I} + v_{NLP})$$

**Calculation:**

1. **Sum the vectors:**
   $$v_{I} + v_{NLP} = [0.1, 0.3] + [0.4, 0.1] = [0.5, 0.4]$$

2. **Average the result:**
   $$h = \frac{1}{2}([0.5, 0.4])$$
   $$h = [0.25, 0.2]$$

---

**Result:**
The resulting hidden vector is **$h = [0.25, 0.2]$**.

#### Step 4: Define Output Matrix ($W'$)

The output weight matrix $W'$ maps the hidden vector back into the vocabulary space.

$$W' = \begin{bmatrix} 0.2 & 0.1 & 0.0 & 0.3 & 0.2 \\ 0.0 & 0.4 & 0.1 & 0.1 & 0.3 \end{bmatrix}$$

**Shape:** $W' \in \mathbb{R}^{2 \times 5}$

---

#### Step 5: Compute Logits ($u = hW'$)

We multiply the hidden vector $h = [0.25, 0.2]$ by the output matrix to get the scores (logits) for each word in the vocabulary.

$$u = [0.25, 0.2] \begin{bmatrix} 0.2 & 0.1 & 0.0 & 0.3 & 0.2 \\ 0.0 & 0.4 & 0.1 & 0.1 & 0.3 \end{bmatrix}$$

**Component Calculations:**

- **$u_0$ (I):** $0.25(0.2) + 0.2(0.0) = 0.05$
- **$u_1$ (love):** $0.25(0.1) + 0.2(0.4) = 0.105$
- **$u_2$ (NLP):** $0.25(0.0) + 0.2(0.1) = 0.02$
- **$u_3$ (is):** $0.25(0.3) + 0.2(0.1) = 0.095$
- **$u_4$ (fun):** $0.25(0.2) + 0.2(0.3) = 0.11$

**Resulting Logit Vector:**
$$u = [0.05, 0.105, 0.02, 0.095, 0.11]$$

---

#### Step 6: Apply Softmax

To convert logits into a probability distribution, we use the Softmax function:
$$y_j = \frac{e^{u_j}}{\sum_{k=1}^{5} e^{u_k}}$$

**1. Compute Exponentials ($e^{u}$):**

- $e^{0.05} \approx 1.0513$
- $e^{0.105} \approx 1.1107$
- $e^{0.02} \approx 1.0202$
- $e^{0.095} \approx 1.0997$
- $e^{0.11} \approx 1.1163$

**2. Sum of Exponentials ($S$):**
$$S = 1.0513 + 1.1107 + 1.0202 + 1.0997 + 1.1163 \approx 5.3982$$

**3. Final Probabilities ($y$):**

- $y_0 \approx 0.1947$
- $y_1 \approx 0.2058$
- $y_2 \approx 0.1890$
- $y_3 \approx 0.2037$
- $y_4 \approx 0.2068$

**Output Distribution:**
$$y \approx [0.195, 0.206, 0.189, 0.204, 0.207]$$

#### Step 7: Interpretation of Results

The model predicts the most likely target word by identifying the index with the **highest probability** in the output vector $y$.

**Top Candidates:**

1.  **fun:** $\approx 0.207$ (Index 4)
2.  **love:** $\approx 0.206$ (Index 1)

**Analysis:**
Currently, the model is slightly favoring "fun" over the actual target word "love." This is expected because we are using **randomly initialized weights**. The probabilities are very close (near $0.20$, which is $1/V$ for our vocabulary size of 5), indicating the model hasn't learned any specific patterns yet.

---

#### Step 8: The Training Process (Backpropagation)

To improve the model, we perform training using a **Loss Function** (typically Cross-Entropy Loss) and **Backpropagation**:

1.  **Calculate Error:** Compare the predicted distribution $y$ against the ground truth (One-Hot vector for "love").
2.  **Update Weights:** Use Gradient Descent to update the matrices $W$ and $W'$.
3.  **Goal:** Adjust the weights so that in future passes, the logit $u_1$ increases, eventually making "love" the clear winner with the highest probability.

## Skip-Gram

While CBOW predicts a **target word** from its context, **Skip-Gram** does the exact opposite: it uses a **single target word** to predict the **surrounding context words**.

### Example

**Sentence:**

> “I love natural language processing”

Take **"natural"** as the target word with window size = 2.

**Context words to predict:**

- `"I"`, `"love"`, `"language"`, `"processing"`

**Training pairs:**
In Skip-Gram, we create individual pairs of `(target, context)`:

1. `("natural", "I")`
2. `("natural", "love")`
3. `("natural", "language")`
4. `("natural", "processing")`

---

### Skip-Gram Internals: Step-by-Step

#### Step 0: Define Vocabulary

| Word | Index |
| :--- | :---- |
| I    | 0     |
| love | 1     |
| NLP  | 2     |
| is   | 3     |
| fun  | 4     |

**Parameters:**

- Vocabulary size ($V$): 5
- Embedding size ($D$): 2
- **Input (Target word):** "love" (Index 1)
- **Output (Expected context words):** "I" (Index 0) and "NLP" (Index 2)

#### Step 1: Define Embedding Matrix ($W$)

$$W = \begin{bmatrix} 0.1 & 0.3 \\ 0.0 & 0.2 \\ 0.4 & 0.1 \\ 0.2 & 0.0 \\ 0.3 & 0.5 \end{bmatrix}$$

#### Step 2: Get Embedding for the Target Word

In Skip-Gram, the hidden vector ($h$) is simply the embedding of the single input word. No averaging is required.

**Target = "love" (Index 1)**
$$h = v_{love} = [0.0, 0.2]$$

#### Step 3: Define Output Matrix ($W'$)

$$W' = \begin{bmatrix} 0.2 & 0.1 & 0.0 & 0.3 & 0.2 \\ 0.0 & 0.4 & 0.1 & 0.1 & 0.3 \end{bmatrix}$$

#### Step 4: Compute Logits ($u = hW'$)

We calculate the score for every word in the vocabulary to see how likely it is to appear near "love".

$$u = [0.0, 0.2] \begin{bmatrix} 0.2 & 0.1 & 0.0 & 0.3 & 0.2 \\ 0.0 & 0.4 & 0.1 & 0.1 & 0.3 \end{bmatrix}$$

**Calculations:**

- **$u_0$ (I):** $0.0(0.2) + 0.2(0.0) = 0.0$
- **$u_1$ (love):** $0.0(0.1) + 0.2(0.4) = 0.08$
- **$u_2$ (NLP):** $0.0(0.0) + 0.2(0.1) = 0.02$
- **$u_3$ (is):** $0.0(0.3) + 0.2(0.1) = 0.02$
- **$u_4$ (fun):** $0.0(0.2) + 0.2(0.3) = 0.06$

**Logit Vector:** $$u = [0.0, 0.08, 0.02, 0.02, 0.06]$$

#### Step 5: Apply Softmax

Convert the logits into probabilities $y$.

**1. Exponentials ($e^u$):**

- $e^{0.0} = 1.0$
- $e^{0.08} \approx 1.0833$
- $e^{0.02} \approx 1.0202$
- $e^{0.02} \approx 1.0202$
- $e^{0.06} \approx 1.0618$

**2. Sum ($S$):** $$S \approx 1.0 + 1.0833 + 1.0202 + 1.0202 + 1.0618 = 5.1855$$

**3. Probabilities ($y$):**

- **$y_0$ (I):** $1.0 / 5.1855 \approx \mathbf{0.1928}$
- **$y_1$ (love):** $1.0833 / 5.1855 \approx 0.2089$
- **$y_2$ (NLP):** $1.0202 / 5.1855 \approx \mathbf{0.1967}$
- **$y_3$ (is):** $\approx 0.1967$
- **$y_4$ (fun):** $\approx 0.2048$

---

### Step 6: Interpretation & Training

**Current State:**
The model currently predicts that "love" is its own neighbor with the highest probability ($0.2089$). Our actual target context words, **"I"** and **"NLP"**, have probabilities of $0.1928$ and $0.1967$.

**The Training Goal:**
During backpropagation, the model will update $W$ and $W'$ so that the vector for "love" ($h$) becomes more similar to the output weights ($W'$) of "I" and "NLP". This increases the probability of $y_0$ and $y_2$ when "love" is the input.

---

## Average Word2Vec

**Average Word2Vec** is a simple yet effective technique for converting a sentence or document into a single fixed-length vector.

While standard Word2Vec (CBOW or Skip-Gram) creates vectors for individual words, Average Word2Vec aggregates them so that you can compare entire chunks of text.

### How it Works

The process is straightforward:

1. Take a sentence (e.g., "I love NLP").

2. Retrieve the Word2Vec vector for each word in that sentence.

3. Calculate the mean of these vectors across each dimension.

### The Mathematical Formula

If a sentence $S$ contains words $\{w_1, w_2, \dots, w_n\}$ and each word has a corresponding vector $v$, the resulting sentence vector $V_{avg}$ is calculated as:

$$V_{avg} = \frac{1}{n} \sum_{i=1}^{n} v_{w_i}$$

### Step-by-Step Example: Average Word2Vec

**1. Vocabulary and Vectors ($D=2$)**
Assume we have the following pre-trained word vectors:

- **I:** $v_1 = [0.1, 0.3]$
- **love:** $v_2 = [0.2, 0.8]$
- **NLP:** $v_3 = [0.9, 0.1]$

**2. The Sentence**

> "I love NLP" ($n=3$)

**3. Summation**
First, we sum the components for each dimension:

- **Dimension 1:** $0.1 + 0.2 + 0.9 = 1.2$
- **Dimension 2:** $0.3 + 0.8 + 0.1 = 1.2$

$$\sum v_i = [1.2, 1.2]$$

**4. Averaging ($n=3$)**
Next, we divide the sums by the total number of words in the sentence:

- **Avg Dim 1:** $1.2 / 3 = 0.4$
- **Avg Dim 2:** $1.2 / 3 = 0.4$

**Resulting Sentence Vector:**
$$V_{avg} = [0.4, 0.4]$$
