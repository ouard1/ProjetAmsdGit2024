
# classification.py
import numpy as np
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import classification_report, accuracy_score

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



def split_data(X, y):
    """Split data into training, testing, and validation sets."""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)
    
    # Save validation data using joblib
    joblib.dump((X_val, y_val), 'validation_data.pkl')
    
    return X_train, X_test, X_val, y_train, y_test, y_val

def train_model(X_train, y_train, X_test, y_test, model_type='svm'):
    """Train classification model and evaluate it on the test set."""
    if model_type == 'svm':
        model = SVC(probability=True, random_state=42)
        param_grid = {'C': [1, 10], 'kernel': ['linear']}
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [50, 100], 'max_depth': [5, None]}
    elif model_type == 'logistic_regression':
        model = LogisticRegression(random_state=42, max_iter=200)
        param_grid = {'C': [1, 10], 'solver': ['liblinear']}
    else:
        raise ValueError("Invalid model type specified.")
    
    # Perform Grid Search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for {model_type}: {grid_search.best_params_}")
    
    # Evaluate on the test set
    test_score = grid_search.score(X_test, y_test)
    print(f"Test accuracy for {model_type}: {test_score}")
    
    return grid_search.best_estimator_



def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set and print the performance metrics."""
    predictions = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")


def save_model(model, model_name='classification_model'):
    """Save the trained model to a file."""
    joblib.dump(model, f'{model_name}.pkl')

if __name__ == "__main__":
    # Step 1: Generate and save the data
    df, X, y = save_data()
    
    # Step 2: Split the data into training, testing, and validation sets
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(X, y)
    
    # Step 3: Train the models using Grid Search
    models = ['svm', 'random_forest', 'logistic_regression']
    for model_type in models:
        model = train_model(X_train, y_train, X_test, y_test, model_type)
        save_model(model, model_type)
        
        # Step 4: Evaluate the model on the test set
        print(f"Evaluating {model_type} on the test set...")
        predictions = model.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=False)
        accuracy = accuracy_score(y_test, predictions)
        print(report)
        print(f"Accuracy: {accuracy}")
        
        
    
    print("Models trained, evaluated, and saved successfully.")
