# Project Data Directory

This directory contains the dataset artifacts for the **Agricultural Extension RAG Benchmark** (TRI AI Cohort 10 Competition).

##  Directory Contents

* **`DATA_CARD.md`**: Detailed breakdown of corpus provenance, schema, preprocessing rules, and statistical distributions.
* **`documents.csv`**: 695 structured agronomic extension passages fused across Title, Crop, Source, and Body Text.
* **`test_queries.csv`**: 200 evaluation queries representing real-world smallholder farming inquiries in Sub-Saharan Africa.

## Corpus Access & Reproducibility

1. **Source Data:** Provided via the TRI AI / Kaggle Competition platform (*Agricultural Extension RAG: Smart Retrieval for Farmers*).
2. **Preprocessing Pipeline:** Raw text fields are enriched automatically during execution using `src/preprocess.py`.
3. **Local Setup:** Place `documents.csv` and `test_queries.csv` inside this `data/` folder before running `scripts/run_pipeline.py`.
