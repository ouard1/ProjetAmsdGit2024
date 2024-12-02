
from sklearn.datasets import make_classification
import joblib
import pandas as pd


def generate_data():
    """Generate synthetic classification dataset."""
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=10,
        n_classes=2,
        random_state=42
    )
    return X, y

def save_data():
    """Generate and save the synthetic dataset."""
    X, y = generate_data()
    df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(X.shape[1])])
    df['target'] = y
    
    # Save the dataset using joblib
    joblib.dump((X, y), 'full_dataset.pkl')
    return df, X, y