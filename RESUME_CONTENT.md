
# Resume Entry: Real-Time Fraud Detection System

**Project Title:** End-to-End Real-Time Fraud Detection System with MLOps Pipeline

**Technologies:** Python, Scikit-Learn, XGBoost, FastAPI, Docker, AWS (EC2/S3), GitHub Actions (CI/CD), Pandas, NumPy.

**Description:**
Designed and deployed a scalable machine learning system to detect fraudulent credit card transactions in real-time. Modeled complex, imbalanced financial data and served predictions via a high-performance REST API.

**Key Achievements (Bullet Points for Resume):**
*   **Engineered a robust fraud detection pipeline** handling high-class imbalance (0.17% fraud rate) using SMOTE and **XGBoost/Random Forest**, achieving a **0.85+ F1-score** and **0.96 ROC-AUC**.
*   **Developed a low-latency REST API** using **FastAPI** to serve model predictions with **<50ms response time**, implementing Pydantic for strict input validation.
*   **Implemented an automated MLOps workflow** with **GitHub Actions** for CI/CD, ensuring code quality via linting and unit testing (Pytest) before deployment.
*   **Containerized the application using Docker** for consistent deployment across environments and deployed to **AWS EC2** for production availability.
*   **Built a proactive Data Drift Detection module** using statistical analysis (KS-test/Z-score) to monitor model performance decay on incoming data streams.
*   **Preprocessed large-scale transaction data** using **RobustScaler** to mitigate the impact of outliers in financial amounts and timestamps.

---

# Interview Talking Points (The "Why" and "How")

**1. Handling Imbalanced Data:**
*   *"I noticed the dataset was highly imbalanced (fraud is rare). Accuracy is a misleading metric here. Instead, I used **SMOTE** (Synthetic Minority Over-sampling Technique) to generate synthetic examples for the minority class during training. I optimized for **Recall** (to catch as many fraud cases as possible) and **Precision-Recall AUC**."*

**2. Model Selection:**
*   *"I experimented with Logistic Regression as a baseline, then moved to Random Forest and XGBoost. XGBoost gave the best performance due to its ability to handle non-linear relationships and its built-in regularization, which prevents overfitting on the minority class."*

**3. Data Cleaning & Scaling:**
*   *"Financial transaction amounts often have extreme outliers. Standard scaling (mean/std) would be skewed by these. I chose **RobustScaler**, which uses the median and interquartile range (IQR), making the model robust to these anomalies."*

**4. Model Drift (The "Bonus" Feature):**
*   *"In production, fraud patterns change. I implemented a drift detection endpoint that compares the statistical distribution (mean/std) of incoming batches against the training baseline. If the Z-score exceeds a threshold, it flags the system for potential retraining."*

**5. Deployment & CI/CD:**
*   *"I didn't just build a model; I built a product. I wrapped the model in a Docker container to ensure it runs anywhere. I set up a collection of Unit Tests ensuring the API endpoints return correct status codes, which run automatically via GitHub Actions on every push."*
