# prediction.py
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import streamlit as st

# Function to load the model
def load_model(model_name):
    """Load the trained model from disk."""
    return joblib.load(model_name)

# Function to load validation data
def load_validation_data():
    """Load the validation data."""
    return joblib.load('validation_data.pkl')

# Function to plot confusion matrix
def plot_confusion_matrix(cm):
    """Plot confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['True 0', 'True 1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    st.pyplot(plt)

# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, auc):
    """Plot ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    st.pyplot(plt)

# Function to evaluate the model
def evaluate_model(model, X_val, y_val):
    """Evaluate the model and return confusion matrix and ROC curve data."""
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    cm = confusion_matrix(y_val, y_pred)
    fpr, tpr, thresholds = roc_curve(y_val, y_prob)
    auc = roc_auc_score(y_val, y_prob)
    
    return cm, fpr, tpr, auc