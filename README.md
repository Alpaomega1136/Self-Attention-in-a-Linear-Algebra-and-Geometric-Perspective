# Self-Attention-Based Sentiment Analysis  
**A Linear Algebra and Geometric Perspective**

---

## Overview

This project implements a **self-attention–based deep learning model** for sentiment analysis on movie reviews.  
The model is inspired by the Transformer architecture and is designed to demonstrate how **self-attention operates as a core contextual modeling mechanism**, grounded in **linear algebra and geometric interpretations**.

Unlike Large Language Models (LLMs), this implementation **does not perform text generation or next-token prediction**.  
Instead, it focuses on **sentence-level sentiment classification**, where self-attention is used to aggregate contextual information across tokens.

The primary objective of this project is **educational**: to bridge theoretical concepts from linear algebra—such as **vector spaces, matrix multiplication, dot products, and projections**—with a practical and interpretable implementation of self-attention.

---

## Key Features

- Token-level self-attention mechanism  
- Scaled dot-product attention  
- Multi-head attention (implemented using PyTorch)  
- Learned positional embeddings  
- Attention matrix extraction for interpretability  
- Binary sentiment classification (positive / negative)  
- Evaluation using the IMDB movie review benchmark  

---

This project implements a **Transformer-inspired deep learning classifier**,  
**not** a full-scale Large Language Model (LLM).

---

## Dataset

The model is trained and evaluated using:

**IMDB Dataset of 50K Movie Reviews – Kaggle**

- 50,000 English movie reviews  
- Balanced positive and negative sentiment labels  
- Widely used benchmark for sentiment analysis  

Dataset link:  
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

## Project Structure
```text
project-root/
├── models/
│   ├── imdb_vocab.json
│   └── imdb_self_attn_best.pt
├── src/
│   ├── attention_core.py
│   └── server.py
├── index.html
├── README.md
└── 13524059_Raymond Jonathan DPJ_Makalah ALGEO.pdf
```


---

## How to Run

This project is designed to be executed locally using a Python server.  
The trained model weights are already provided, so **no training step is required**.

### Requirements

Make sure Python is installed and install PyTorch:

```bash
pip install torch
```

### Running the Application

1. Navigate to the project root directory.
2. Run the Python server:

```bash
python src/server.py
```

3. Open a web browser and access the local interface:
```bash
http://localhost:5000
```

4. Enter a movie review sentence to observe:
- Positive sentiment probability
- Sentiment classification result
- Self-attention weight matrix visualization

### Interpretability Focus

In addition to prediction accuracy, this project emphasizes model interpretability.
The self-attention mechanism explicitly produces an attention matrix that represents token-to-token interactions.

By visualizing this matrix, users can observe:
- Which words receive higher attention
- How sentiment-relevant tokens dominate contextual representations
- How attention shifts in long, contrastive, or ambiguous sentences

This aligns with the linear algebraic interpretation of self-attention as a sequence of matrix transformations and similarity-based projections in high-dimensional vector spaces.

### Author
Raymond Jonathan Dwi Putra Julianto
Informatics Engineering
Institut Teknologi Bandung

### License
This project is intended for academic and educational use only.