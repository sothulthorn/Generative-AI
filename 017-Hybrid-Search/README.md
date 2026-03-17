# Hybrid Search RAG with LangChain and Pinecone

## Overview

This project demonstrates **Hybrid Search** in a Retrieval-Augmented Generation (RAG) pipeline using LangChain and Pinecone. Hybrid Search combines multiple search techniques to deliver more accurate and relevant results than either technique alone.

## What is Hybrid Search?

Traditional RAG pipelines rely on **Semantic Search** alone — converting text into dense vectors and finding similar documents via cosine similarity. Hybrid Search improves on this by combining two complementary approaches:

| Technique | How It Works | Representation | Strength |
|---|---|---|---|
| **Semantic Search** | Dense vector search using embeddings | Dense Vectors (e.g., 384-dim float arrays) | Finds contextually similar results even with different wording |
| **Syntactic (Keyword) Search** | Exact/keyword matching using sparse vectors | Sparse Matrix (via TF-IDF, BM25, BoW) | Finds exact keyword matches that semantic search may miss |

### How Hybrid Search Works

1. A document is converted into both **dense vectors** (via embedding models like HuggingFace, OpenAI) and **sparse vectors** (via BM25/TF-IDF).
2. When a user query arrives, it is also encoded into both dense and sparse representations.
3. Two parallel searches are performed:
   - **Vector Search** (semantic) — produces `Result_S`
   - **Keyword Search** (syntactic) — produces `Result_K`
4. Results are merged using **Reciprocal Rank Fusion (RRF)** to produce the final ranked output.
5. The top results are passed to an LLM for response generation.

## Reciprocal Rank Fusion (RRF)

RRF is the algorithm used to combine rankings from multiple search results. It computes a final score for each document using:

```
Final Score = Sum of 1 / (C + rank_d)
```

Where:
- `C` is a constant (database-dependent, typically 1 to 60)
- `rank_d` is the document's rank in a given result list

**Example (C = 0):**

| Document | Semantic Rank | Keyword Rank | Score Calculation | Final Score |
|---|---|---|---|---|
| Document 1 | 1 (score 1.0) | 5 (score 0.2) | 1 + 0.2 | **1.2** |
| Document 2 | 2 (score 0.5) | 3 (score 0.33) | 0.5 + 0.33 | **0.83** |
| Document 3 | 3 (score 0.33) | 2 (score 0.5) | 0.33 + 0.5 | **0.83** |

The weighting between keyword and semantic search can be tuned (e.g., 50/50 or 70/30) depending on the use case.

## Implementation

### Tech Stack

- **LangChain** — orchestration framework
- **Pinecone** — vector database (supports hybrid search with `dotproduct` metric)
- **HuggingFace Embeddings** — `all-MiniLM-L6-v2` model (384 dimensions) for dense vectors
- **BM25Encoder** (pinecone-text) — for sparse vector encoding
- **PineconeHybridSearchRetriever** — LangChain retriever that handles both dense and sparse search

### Key Steps

1. **Create a Pinecone Index** with `dotproduct` metric (required for sparse vector support):
    ```python
    pc.create_index(
        name="hybrid-search-langchain-pinecone",
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    ```

2. **Initialize Dense Embeddings** using HuggingFace:
    ```python
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    ```

3. **Initialize Sparse Encoder** using BM25:
    ```python
    from pinecone_text.sparse import BM25Encoder
    bm25_encoder = BM25Encoder().default()
    bm25_encoder.fit(sentences)
    bm25_encoder.dump("bm25_values.json")
    ```

4. **Create the Hybrid Search Retriever**:
    ```python
    from langchain_community.retrievers import PineconeHybridSearchRetriever
    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25_encoder,
        index=index
    )
    ```

5. **Add documents and query**:
    ```python
    retriever.add_texts(["In 2023, I visited Paris", ...])
    retriever.invoke("What city did I visit first?")
    ```

## Graph Knowledge (Bonus Concept)

The PDF also introduces **Graph Knowledge Search** as an advanced extension. Using a Graph Database like **Neo4j**, a RAG application can combine three search methods:

- Keyword Search
- Semantic Search
- Graph Knowledge Search (leveraging entity relationships)

## Setup

1. Install dependencies:
    ```bash
    pip install pinecone-client pinecone-text pinecone-notebooks langchain langchain-community langchain_huggingface sentence_transformers python-dotenv
    ```

2. Set up environment variables:
    ```
    HF_TOKEN=<your_huggingface_token>
    ```

3. Set your Pinecone API key in the notebook or environment.

4. Run the `experiments.ipynb` notebook.

## Project Structure

```
017-Hybrid-Search/
├── experiments.ipynb              # Main notebook with hybrid search implementation
├── bm25_values.json               # Pre-fitted BM25 sparse encoder values
├── 001 hybridsearch-final.pdf     # Reference notes/diagrams on hybrid search
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```
