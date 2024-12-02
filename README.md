
---
title: "Model Evaluation and Visualization App"
author: "Ouarda Boumansour & Ala Eddine Choulr-Allah"
date: "`r Sys.Date()`"
output: github_document
---

# Model Evaluation and Visualization App

This repository contains a machine learning project for training, evaluating, and visualizing classification models. The project includes scripts for data generation, model training, evaluation, and a Streamlit-based web app for interactive visualization of results. Additionally, the project is containerized using Docker for consistent and portable deployment.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Docker Setup](#docker-setup)
- [Technologies Used](#technologies-used)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

## Overview

This project demonstrates a complete workflow for building and evaluating machine learning classification models:

- **Data generation**: Synthetic dataset generation using `classification.py`.
- **Model training**: Training multiple models (SVM, Random Forest, Logistic Regression) with hyperparameter tuning.
- **Evaluation and visualization**: Generating performance metrics (Confusion Matrix, ROC Curve) and visualizations using `Streamlit` in `app.py`.

The app is containerized using Docker for easy deployment and reproducibility.

## Features

1. **Synthetic Data Generation**:
   - Creates a classification dataset with configurable features and saves it locally.

2. **Model Training**:
   - Trains multiple models with hyperparameter tuning using `GridSearchCV`.
   - Models supported:
     - Support Vector Machine (SVM)
     - Random Forest
     - Logistic Regression

3. **Interactive Web App**:
   - Visualizes dataset features using PCA and correlation heatmaps.
   - Displays evaluation metrics, Confusion Matrix, and ROC Curve for the selected model.
   - Allows users to interactively select models for evaluation.

4. **Dockerized Application**:
   - Deployable as a container for consistent and portable execution.

## Project Structure

```plaintext
.
├── classification.py          # Script for data generation and model training
├── prediction.py              # Script for model evaluation and prediction
├── app.py                     # Streamlit app for interactive visualization
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Dockerfile for containerizing the app
├── docker-compose.yml         # Docker Compose file for easy deployment
└── README.md                  # Project documentation
```

## Installation

### Prerequisites

- Python 3.9+
- Docker (for containerization)
- Optional: `virtualenv` for Python environment isolation

### Local Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/ouard1/ProjetAmsdGit2024.git
   cd model-evaluation-app
   ```



2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Generate the dataset and train the models by running classification.py:

    ```bash
    python classification.py
    ```
4. Run the app locally:

   ```bash
   streamlit run app.py
   ```

## Usage

1. **Generate Synthetic Data**:

   Run `classification.py` to generate the dataset and train models:

   ```bash
   python classification.py
   ```

2. **Evaluate Models**:

   Use the Streamlit app (`app.py`) to evaluate trained models interactively:

   ```bash
   streamlit run app.py
   ```

3. **Modify Scripts**:

   Update parameters in `classification.py` or `app.py` to customize data generation, training, and evaluation.

## Docker Setup

### Build and Run the Docker Container

1. Build the Docker image:

   ```bash
   docker build -t model-evaluation-app .
   ```

2. Run the container:

   ```bash
   docker run -p 8501:8501 model-evaluation-app
   ```

### Using Docker Compose

1. Run the app with `docker-compose`:

   ```bash
   docker-compose up --build
   ```

2. Access the app at `http://localhost:8501`.

### Push to Docker Hub

1. Tag the image:

   ```bash
   docker tag model-evaluation-app <your_dockerhub_username>/model-evaluation-app
   ```

2. Push the image:

   ```bash
   docker push <your_dockerhub_username>/model-evaluation-app
   ```

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - Machine Learning: `scikit-learn`, `joblib`
  - Data Visualization: `matplotlib`, `seaborn`, `pandas`
  - Interactive Web Apps: `Streamlit`
- **Containerization**: Docker, Docker Compose



## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any changes or improvements.

## Contact

For questions or support, please contact:

- Name: Ouarda Boumansour & Ala Eddine Choukr-Allah
- Email: Alaeddinechoukr@gmail.com


---

