# Support Integrity Auditor (SIA)

An AI-powered auditing system designed for enterprise CRM platforms that automatically detects discrepancies between assigned ticket priorities and their actual severity. By combining semantic understanding, rule-based reasoning, and operational signals, SIA identifies hidden critical issues and inflated priority assignments before they impact customer experience and SLA compliance.

**Live Demo:**
https://support-integrity-auditor-anuj11.streamlit.app/

---

# Features

* Automated Priority Mismatch Detection
* Self-Supervised Pseudo-Label Generation
* Multi-Signal Severity Assessment
* Fine-Tuned Transformer Classification
* Evidence-Based Audit Reports
* Batch CSV Processing
* Interactive Streamlit Dashboard
* Explainable AI Workflow

---

# System Architecture

> Save your architecture screenshot as **architecture.png** in the repository root and use:

```markdown
![System Architecture](architecture.png)
```

### Pipeline Overview

The Support Integrity Auditor follows a three-stage workflow:

### Stage 1 — Severity Estimation

Raw customer support tickets undergo preprocessing and feature extraction. Three independent severity indicators are generated:

* Semantic Urgency Analysis (all-MiniLM-L6-v2)
* Rule-Based Escalation Detection
* Resolution-Time Regression Modeling

These signals are fused into a single severity estimate and calibrated into discrete priority levels.

### Stage 2 — Mismatch Classification

Pseudo-labels generated during Stage 1 are combined with ticket metadata and descriptions to train a compact BERT-based classifier using weighted cross-entropy loss.

### Stage 3 — Audit & Deployment

The trained model powers:

* Batch inference through `predict.py`
* Interactive Streamlit dashboard
* Evidence dossier generation for flagged tickets

---

# Methodology

## Self-Supervised Severity Inference

Since the dataset does not contain explicit mismatch annotations, the system constructs its own supervision signal using multiple independent severity indicators.

### Semantic Severity Score (S_sem)

The system uses **all-MiniLM-L6-v2** embeddings to measure similarity between ticket content and predefined urgency concepts. This enables contextual severity estimation beyond simple keyword matching.

### Rule-Based Severity Score (S_rule)

A lightweight NLP engine identifies urgency indicators, escalation phrases, legal threats, refund requests, and service-failure terminology.

### Resolution-Time Severity Score (S_res)

A Ridge Regression model predicts expected resolution duration from textual ticket information, allowing historically complex tickets to contribute to severity estimation.

---

# Severity Fusion Strategy

The three severity indicators are normalized and combined using a weighted scoring mechanism:

```text
Fused Score = (0.5 × S_sem) + (0.3 × S_rule) + (0.2 × S_res)
```

The fused score is calibrated to match the original priority distribution using percentile thresholds.

Priority levels:

* Low
* Medium
* High
* Critical

A ticket is flagged as a mismatch when the difference between assigned and inferred severity is at least two priority levels.

### Hidden Crisis

High-severity ticket assigned a Low or Medium priority.

### False Alarm

Low-severity ticket assigned a High or Critical priority.

---

# Signal Validation & Ablation Study

The effectiveness of the pseudo-label generation process is validated through pairwise agreement analysis.

| Signal Configuration                    | Agreement Rate    | Contribution                                                |
| --------------------------------------- | ----------------- | ----------------------------------------------------------- |
| Semantic Urgency vs Rule-Based Features | 78.4%             | Strong alignment while capturing different urgency patterns |
| Semantic Severity (S_sem)               | Baseline          | Context-aware semantic understanding                        |
| Rule-Based NLP (S_rule)                 | Supporting Signal | Explicit escalation and urgency detection                   |
| Resolution-Time Proxy (S_res)           | Auxiliary Signal  | Complexity-driven severity estimation                       |
| SIA Fused Pipeline                      | Reference System  | Unified pseudo-label generation framework                   |

---

# Model Training

The mismatch detector is based on:

```text
google/bert_uncased_L-2_H-128_A-2
```

Training characteristics:

* Fine-tuned on pseudo-labeled data
* Weighted cross-entropy loss
* CPU-friendly architecture
* Handles approximately 80:20 class imbalance

Input features include:

* Ticket Description
* Ticket Subject
* Ticket Channel
* Resolution Signals
* Metadata Features

---

# Performance Results

Evaluation was conducted on a held-out 20% test split.

| Metric                         | Required Threshold | Achieved Score |   |
| ------------------------------ | ------------------ | -------------- | - |
| Binary Classification Accuracy | ≥ 83%              | 89.7%          |   |
| Macro F1 Score                 | ≥ 0.82             | 0.85           |   |
| Recall (Consistent Class)      | ≥ 0.78             | 0.9            |   |
| Recall (Mismatch Class)        | ≥ 0.78             | 0.88           |   |

The model exceeds all required verification criteria.

---

# Repository Structure

```text
.
├── app.py
├── train_pipeline.py
├── predict.py
├── notebook.ipynb
├── requirements.txt
├── saved_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── sample_tickets.csv
├── sample_audited.csv
└── sample_dossiers.json
```

---

# Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

# Training Pipeline

Run the complete training workflow:

```bash
python train_pipeline.py
```

This process:

* Generates pseudo-labels
* Trains the classifier
* Saves model checkpoints
* Prepares inference artifacts

Saved models are stored in:

```text
saved_model/
```

---

# Batch Auditing

Audit support tickets using:

```bash
python predict.py --input enhanced_customer_support_data.csv --output audited_results.csv --dossier evidence_dossiers.json
```

Generated outputs:

### audited_results.csv

* Predictions
* Severity estimates
* Mismatch labels

### evidence_dossiers.json

* Structured audit explanations
* Supporting evidence signals
* Confidence information

---

# Launch Web Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

Dashboard capabilities:

* Single Ticket Analysis
* Batch CSV Upload
* Priority Mismatch Detection
* Evidence Dossier Viewer
* Audit Statistics Dashboard
* Severity Distribution Insights

