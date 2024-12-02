import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from predict_classification import load_model, load_validation_data, evaluate_model, plot_confusion_matrix, plot_roc_curve

def load_full_dataset():
    """Load the full dataset generated in classification.py."""
    X, y = joblib.load('full_dataset.pkl')
    return X, y

def plot_pca_with_class_separation(X, y):
    """Perform PCA on the entire dataset and plot the first two principal components with class separation."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(12, 8))  # Increase the figure size for a wider plot

    # Scatter plot with colors based on the class labels
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    
    # Add a color bar for class labels
    cbar = plt.colorbar(scatter)
    cbar.set_label('Class Label')

    ax.set_title('PCA of Full Dataset (Class Separation)', fontsize=16)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)

    return fig

def plot_correlation_heatmap(X):
    """Plot a correlation heatmap to show relationships between features."""
    # Convert to a DataFrame for correlation matrix computation
    df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(X.shape[1])])
    
    # Compute the correlation matrix
    corr = df.corr()

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 10))  # Increased size for better visibility
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
    ax.set_title('Correlation Heatmap of Features', fontsize=16)
    return fig



def main():
    st.title("Model Evaluation App")
    st.subheader("Data Visualisation for our dataset")  # Added title above the model selection

    # Load the full dataset
    X, y = load_full_dataset()

    # Create a 2x2 grid for visualizations in the first row
    col1, col2 = st.columns([3, 3])  # Set equal width columns (adjust ratio if needed)
    row1_col1, row1_col2 = col1, col2

 

    # First Row - PCA with class separation and Correlation Heatmap
    with row1_col1:
        st.subheader('PCA - First Two Principal Components (Class Separation)')
        pca_fig = plot_pca_with_class_separation(X, y)
        st.pyplot(pca_fig)

    with row1_col2:
        st.subheader('Correlation Heatmap')
        heatmap_fig = plot_correlation_heatmap(X)
        st.pyplot(heatmap_fig)

   

    # Model selection
    st.subheader("Let's predict , choose a model")
    model_type = st.selectbox("Select Model", ('svm', 'random_forest', 'logistic_regression'))
    model = load_model(f'{model_type}.pkl')

    # Load validation data
    X_val, y_val = load_validation_data()

    # Evaluate the model
    cm, fpr, tpr, auc = evaluate_model(model, X_val, y_val)

   
    col3, col4 = st.columns([3, 3])  # Set equal width columns (adjust ratio if needed)
    row1_col3, row1_col4 = col3, col4

    with row1_col4:
    # Display confusion matrix
        st.subheader('Confusion Matrix')
        plot_confusion_matrix(cm)

    # Display ROC curve
    
    with row1_col3:
        st.subheader('ROC Curve')
        plot_roc_curve(fpr, tpr, auc)

    # Display accuracy and AUC
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
    st.write(f"ROC AUC: {auc:.2f}")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Model evaluation app",   # App title in browser tab
        page_icon="π",          # Emoji or path to icon file
        layout="wide",                    # "centered" or "wide"
        initial_sidebar_state="collapsed" # "auto", "expanded", "collapsed"
    )
    main()
