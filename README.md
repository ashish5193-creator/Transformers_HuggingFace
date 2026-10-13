# Sarcasm Detection using Hugging Face Transformers

This project demonstrates how to fine-tune a pre-trained RoBERTa model for sarcasm detection on a Hindi-English (Hinglish) social media dataset using the Hugging Face Transformers ecosystem.

The project covers the complete machine learning workflow:

- Dataset Loading
- Data Exploration
- Label Encoding
- Tokenization
- Dataset Splitting
- Model Fine-Tuning
- Evaluation
- Inference
- Error Analysis
- Hugging Face Pipelines Demonstration

---

# Project Overview

Sarcasm detection is a challenging Natural Language Processing (NLP) task because sarcastic statements often express the opposite of their literal meaning.

Example:

```text
"Bahut badhiya, salary fir delay ho gayi"
```

Literal Meaning:

```text
Positive
```

Actual Meaning:

```text
Negative / Sarcastic
```

The goal of this project is to train a transformer-based classifier capable of detecting sarcasm in Hinglish social media text.

---

# Dataset

Dataset Source:

```python
load_dataset("HuggMachas/Sarcasm_dataset")
```

Dataset contains:

```text
Text
Label
Language Tokens
Tweet Metadata
```

Example:

```json
{
  "text": "Triple Talaq par Burbak Kuchh nahi bolega",
  "label": "NO"
}
```

Labels:

| Original | Encoded |
|----------|----------|
| NO | 0 |
| YES | 1 |

Where:

```text
0 = Not Sarcastic
1 = Sarcastic
```

---

# Dataset Statistics

Total Samples:

```text
5250
```

Class Distribution:

```text
Not Sarcastic : 4746
Sarcastic     : 504
```

Distribution:

```text
90.4% Non-Sarcastic
9.6% Sarcastic
```

The dataset is highly imbalanced.

---

# Technology Stack

- Python
- Hugging Face Transformers
- Hugging Face Datasets
- PyTorch
- Accelerate
- NumPy

---

# Installation

Install dependencies:

```bash
pip install transformers datasets accelerate evaluate torch
```

---

# Load Dataset

```python
from datasets import load_dataset

dataset = load_dataset(
    "HuggMachas/Sarcasm_dataset"
)
```

---

# Data Preprocessing

Convert labels to binary values:

```python
def convert_labels(example):

    example["label"] = (
        1 if example["label"] == "YES"
        else 0
    )

    return example
```

Apply transformation:

```python
dataset = dataset.map(convert_labels)
```

---

# Model Selection

Pretrained Model:

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
```

Tokenizer:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)
```

Model:

```python
from transformers import (
    AutoModelForSequenceClassification
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    ignore_mismatched_sizes=True
)
```

Why RoBERTa?

- Strong contextual representation
- Robust performance on social media text
- Better handling of informal language
- Transfer learning capability

---

# Tokenization

```python
def tokenize(example):

    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
```

Apply:

```python
tokenized_dataset = dataset.map(
    tokenize,
    batched=True
)
```

---

# Train/Test Split

```python
split_dataset = (
    tokenized_dataset["train"]
    .train_test_split(
        test_size=0.2,
        seed=42
    )
)
```

Result:

```text
Train : 4200 samples
Test  : 1050 samples
```

---

# Training Configuration

```python
TrainingArguments(
    output_dir="./sarcasm_model",

    evaluation_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    num_train_epochs=3,

    weight_decay=0.01,

    logging_steps=50,

    load_best_model_at_end=True,

    report_to="none"
)
```

---

# Model Training

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"]
)

trainer.train()
```

Training Summary:

```text
Epoch 1
Train Loss      0.128579
Validation Loss 0.125632

Epoch 2
Train Loss      0.131983
Validation Loss 0.130633

Epoch 3
Train Loss      0.074151
Validation Loss 0.141239
```

Final Training Metrics:

```text
Training Loss: 0.1261
Training Runtime: 334 seconds
```

---

# GPU Utilization

Model trained using CUDA:

```python
print(model.device)
```

Output:

```text
cuda:0
```

---

# Inference

Example:

```python
text = "वाह! फिर से इंटरनेट बंद हो गया, क्या शानदार सेवा है"
```

Prediction:

```text
Not Sarcastic
```

Probability:

```text
0.9977
```

---

# Prediction Workflow

```text
Input Text
      |
Tokenizer
      |
Input IDs
      |
RoBERTa Model
      |
Logits
      |
Softmax
      |
Prediction
```

---

# Testing Custom Examples

Sarcastic Examples:

```text
Wah kya baat hai, 6 ghante train late hai #Irony
```

Prediction:

```text
Sarcastic 😏
Confidence: 77.7%
```

---

```text
Maiden shabash bas ab aise hi khelo #Sarcasm
```

Prediction:

```text
Sarcastic 😏
Confidence: 73.5%
```

---

```text
Bahut badhiya banking service,
3 din se server down hai #Irony
```

Prediction:

```text
Sarcastic 😏
Confidence: 78.0%
```

---

# Normal Examples

```text
Aaj mausam bahut accha hai
```

Prediction:

```text
Not Sarcastic 🙂
Confidence: 99.8%
```

---

```text
Mujhe cricket pasand hai
```

Prediction:

```text
Not Sarcastic 🙂
Confidence: 99.8%
```

---

# Error Analysis

The model performs well when explicit cues exist:

Examples:

```text
#Irony
#Sarcasm
```

However it struggles with implicit sarcasm:

Example:

```text
Bahut badhiya,
salary fir delay ho gayi
```

Prediction:

```text
Not Sarcastic
```

Reason:

- No explicit sarcasm marker
- Less contextual information
- Dataset bias toward hashtag-based sarcasm

---

# Evaluation

Predictions on Test Set:

```text
Class 0 (Non-Sarcastic):
910

Class 1 (Sarcastic):
140
```

The model learns sarcasm patterns but heavily favors the majority class due to dataset imbalance.

---

# Understanding the Imbalance Problem

Training Distribution:

```text
Sarcastic:
504

Non-Sarcastic:
4746
```

Ratio:

```text
1 : 9.4
```

This causes:

- High accuracy
- Lower sarcasm recall
- Difficulty identifying subtle sarcasm

Potential Solutions:

- Class weights
- Weighted loss functions
- Oversampling sarcasm examples
- Data augmentation
- Focal loss

---

# Hugging Face Pipelines Demonstration

The notebook also demonstrates several built-in Hugging Face pipelines.

---

## 1. Sentiment Analysis

```python
pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
```

Example:

```text
I love using Hugging Face transformers!
```

Output:

```text
POSITIVE
Score: 0.997
```

---

## 2.
