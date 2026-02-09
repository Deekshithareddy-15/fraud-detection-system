# Fraud Detection System

A production-grade, Cloud-Native Fraud Detection System using Machine Learning and FastAPI.

## Project Structure

- `data/`: Raw and processed datasets
- `models/`: Trained models and evaluation metrics
- `src/`: Source code for ML pipeline (ingestion, training, evaluation)
- `api/`: FastAPI backend setup
- `notebooks/`: Jupyter notebooks for EDA
- `tests/`: Unit tests

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate/Download Data:
   ```bash
   python src/data_ingestion.py
   ```

3. Train Model:
   ```bash
   python src/train_pipeline.py
   ```

4. Run API:
   ```bash
   uvicorn api.main:app --reload
   ```
