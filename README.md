# Self-Attention-Based Sentiment Analysis  
**A Linear Algebra and Geometric Perspective**

## Overview
This project implements a **self-attention–based deep learning model** for sentiment analysis on movie reviews.  
The model is inspired by the Transformer architecture and focuses on demonstrating how **self-attention operates as a core contextual modeling mechanism**, grounded in **linear algebra and geometric interpretations**.

Unlike Large Language Models (LLMs), this implementation is **not designed for text generation or next-token prediction**, but rather for **sentence-level sentiment classification**.  
The primary objective is educational: to bridge **theoretical concepts in linear algebra** (vectors, matrices, dot products, projections) with a **practical self-attention implementation**.

---

## Key Features
- Token-level **self-attention mechanism**
- Scaled dot-product attention
- Multi-head attention (via PyTorch)
- Positional embeddings
- Attention matrix extraction for interpretability
- Sentiment classification on IMDB movie reviews

---

## Model Category
- **Artificial Intelligence**: ✅  
- **Machine Learning**: ✅  
- **Deep Learning**: ✅  
- **Large Language Model (LLM)**: ❌  

This project implements a **Transformer-inspired deep learning classifier**, not a full-scale LLM.

---

## Dataset
The model is trained and evaluated using:

**IMDB Dataset of 50K Movie Reviews – Kaggle**  
- 50,000 movie reviews  
- Balanced positive and negative sentiment labels  
- Widely used benchmark for sentiment analysis  

Dataset link:  
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

## Project Structure
.
├── models/
│ ├── imdb_vocab.json
│ └── imdb_self_attn_best.pt
├── src/
│ ├── attention_core.py
│ └── server.py
├── index.html
├── README.md
└── Makalah_Algeo.pdf


---

Author

Raymond Jonathan Dwi Putra Julianto
Informatics Engineering
Institut Teknologi Bandung

License

This project is intended for academic and educational use only.