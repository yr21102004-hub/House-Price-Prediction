# 🏠 House Price Analysis & Prediction

A modularized Machine Learning project that predicts house prices using Linear Regression. This project features a synthetic data generator, model training pipeline, and an interactive web dashboard built with Streamlit.

## 🚀 Overview

This project is divided into three main modules, each handled by a specific logic component:
1.  **Data Preprocessing**: Generates realistic synthetic housing data and handles feature encoding.
2.  **Model Training**: Implements a Linear Regression model using Scikit-Learn, evaluates performance, and extracts feature importance.
3.  **Interactive UI**: A Streamlit-based web application for data visualization, model evaluation, and real-time price prediction.

## 🛠️ Features

-   **Synthetic Data Generation**: Creates a dataset of 200 apartments with features like Area, Bedrooms, Age, and Location.
-   **Machine Learning Pipeline**: 
    -   Feature Engineering (One-Hot Encoding for locations).
    -   Data Splitting (80% Train / 20% Test).
    -   Model: `LinearRegression`.
-   **Interactive Dashboard**:
    -   **Data Explorer**: View the raw dataset and correlation scatter plots.
    -   **Model Metrics**: Real-time display of MAE and R² Score.
    -   **Prediction Engine**: Input house details to get an instant price estimation.

## 📁 Project Structure

-   `student1_data_preprocessing.py`: Handles data generation and preparation.
-   `student2_model_training.py`: Handles model creation, training, and evaluation.
-   `student3_visualization_ui.py`: The main entry point for the Streamlit web application.
-   `Project_Explanation.md`: Detailed technical breakdown of the code (Arabic).

## ⚙️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd House-Price-Analysis-Prediction
    ```

2.  **Install dependencies**:
    ```bash
    pip install pandas numpy scikit-learn streamlit matplotlib seaborn
    ```

## 🖥️ Usage

To run the application, execute the following command in your terminal:

```bash
streamlit run student3_visualization_ui.py
```

## 📊 Model Performance

The model typically achieves high accuracy (R² > 0.90) by learning the underlying pricing formula used in the synthetic data generator, which accounts for area, room count, age, and location premiums.

---
*Created as a collaborative student project for House Price Analysis.*
