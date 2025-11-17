

### Character-Level RNN • Mini Transformer Encoder • Scaled Dot-Product Attention  
**Course:** Natural Language Processing  
**Student:** Vishal Vusnagiri  

---

# 📘 Overview
This programming assignment implements three foundational deep-learning components used in Natural Language Processing:

1. **Character-Level RNN Language Model**  
2. **Mini Transformer Encoder (self-attention only)**  
3. **Scaled Dot-Product Attention**

All implementations are done in **PyTorch**, with outputs including training curves, generated samples, attention matrices, and contextual embeddings.

---

# ============================================================
# 🧩 1 — Character-Level RNN Language Model
# ============================================================

## 🎯 Objective
Train a small RNN/GRU/LSTM-based model to predict the next character in a sequence using teacher forcing.

---

## 📄 Data
Two datasets are used:

### **1. Toy Corpus (manually created)**
Example:
```

hello hello help helios hello help

```

### **2. Real Corpus (50–200 KB)**
Any public-domain file such as:
- Sherlock Holmes (Gutenberg)
- Shakespeare text
- Wikipedia article dump

Saved as:
```

data.txt

```

---

## 🧠 Model Architecture
```

Character Embedding Layer
↓
RNN / GRU / LSTM (Hidden size = 64–256)
↓
Fully Connected Layer
↓
Softmax over characters

```

### Training Setup
| Parameter | Value |
|----------|-------|
| Sequence length | 50–100 |
| Batch size | 64 |
| Hidden size | 64–256 |
| Epochs | 5–20 |
| Optimizer | Adam |
| Loss | Cross-Entropy |
| Method | Teacher forcing |

---

## 📊 Required Outputs
1. **Training + validation loss curves**  
   → `loss_curve.png`
2. **Three text generations** (200–400 chars each):  
   - τ = **0.7**  
   - τ = **1.0**  
   - τ = **1.2**
3. **Reflection (3–5 sentences)** about:
   - Sequence length  
   - Hidden size  
   - Temperature  

---


```

---

# ============================================================
# 🧩 2 — Mini Transformer Encoder
# ============================================================

## 🎯 Objective
Implement a simplified **Transformer Encoder** block (not a decoder) to contextualize word embeddings.

---

## 📄 Dataset
Create **10 short sentences**, e.g.:

```

["hello world",
"attention is powerful",
"pytorch is fun",
"transformers learn context",
"deep learning models",
"sequence modeling tasks",
...]

```

---

## 🧠 Components Implemented
- Tokenization  
- Word embeddings  
- **Sinusoidal positional encoding**  
- **Scaled dot-product attention**  
- **Multi-head attention (2–4 heads)**  
- Feed-forward network  
- Residual connections  
- Layer normalization  

---

## 📊 Required Outputs
- **Tokenized input**
- **Contextual embeddings** after Transformer encoder
- **Attention heatmap** saved as:  
  `attention_heatmap.png`

---


```

---

# ============================================================
# 🧩 3 — Scaled Dot-Product Attention
# ============================================================

## 🎯 Objective
Implement and test the core attention formula:

\[
\text{Attention}(Q, K, V) =
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

---

## 🧪 Requirements
Your program must:

- Generate random matrices Q, K, V  
- Compute:
  - Raw attention scores `QK^T`
  - Scaled scores `QK^T / sqrt(d_k)`
  - Softmax attention weights
  - Final output vectors  
- Show a **numerical stability comparison** (before vs. after scaling)

---

