
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import os

def generate_synthetic_data(n_samples=100000, output_path="data/raw/creditcard.csv"):
    """
    Generates a synthetic dataset mimicking credit card fraud data.
    """
    print(f"Generating synthetic data with {n_samples} samples...")
    
    # 28 PCA components (V1-V28), Time, Amount
    n_features = 30 
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.99, 0.01], # 1% fraud
        random_state=42
    )
    
    columns = [f"V{i+1}" for i in range(28)] + ["Time", "Amount"]
    
    # Adjust Time and Amount to look more realistic
    # (Note: make_classification returns centered data, so we shift it)
    X[:, -2] = np.abs(X[:, -2] * 1000 + 10000) # Time
    X[:, -1] = np.abs(X[:, -1] * 100) # Amount
    
    df = pd.DataFrame(X, columns=columns)
    df["Class"] = y
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(f"Class distribution:\n{df['Class'].value_counts(normalize=True)}")

if __name__ == "__main__":
    generate_synthetic_data()
