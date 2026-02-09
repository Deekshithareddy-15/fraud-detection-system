
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def clean_data(df):
    """Performs basic data cleaning."""
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    new_rows = len(df)
    if initial_rows != new_rows:
        print(f"Removed {initial_rows - new_rows} duplicate rows.")
    
    # Handle missing values if any (though typical fraud datasets are clean)
    if df.isnull().sum().sum() > 0:
        print(f"Dataset contains missing values. Imputing with median.")
        df = df.fillna(df.median())
        
    return df

def preprocess_features(df):
    """Scales Amount and Time features."""
    # RobustScaler is less prone to outliers.
    scaler_amount = RobustScaler()
    scaler_time = RobustScaler()

    df['scaled_amount'] = scaler_amount.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler_time.fit_transform(df['Time'].values.reshape(-1, 1))

    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    return df, scaler_amount, scaler_time

def split_data(df, target_col='Class', test_size=0.2, random_state=42):
    """Splits data into training and testing sets."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
