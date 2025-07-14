# Real Estate Price Prediction

## Overview

This project analyzes and predicts real estate unit prices using various machine learning regression techniques. The workflow includes data preprocessing, exploratory data analysis, feature engineering, and the application of multiple regression models (Linear Regression, Random Forest, PCA-based regression, and Polynomial Regression). The project is implemented in Python using Jupyter Notebook.

---

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset Description](#dataset-description)
- [Setup & Requirements](#setup--requirements)
- [How to Run](#how-to-run)
- [Analysis & Modeling Steps](#analysis--modeling-steps)
- [Results & Evaluation](#results--evaluation)
- [Outputs](#outputs)
- [References](#references)

---

## Project Structure

```
.
├── Source_code_(SSANAH_Group).ipynb   # Main Jupyter notebook with all code and analysis
├── Real estate.xlsx                   # Original dataset
├── Riyalstate.xlsx                    # Processed dataset (after feature engineering)
├── ES304 CEP Report (SSANAH Group).docx # Project report (optional)
```

---

## Dataset Description

The dataset (`Real estate.xlsx`) contains real estate transaction data with the following features:

- **X1 transaction date**: Transaction date (float, e.g., 2012.917)
- **X2 house age**: Age of the house (years)
- **X3 distance to the nearest MRT station**: Distance (meters)
- **X4 number of convenience stores**: Count of stores nearby
- **X5 latitude**: Latitude coordinate (removed after feature engineering)
- **X6 longitude**: Longitude coordinate (removed after feature engineering)
- **No**: Row number (removed)
- **Y house price of unit area**: Target variable (price per unit area)

**Feature Engineering:**
- A new feature, **X5 Vector Mag**, is created as the Euclidean magnitude of latitude and longitude, representing the spatial location as a single value.
- The columns `X5 latitude`, `X6 longitude`, and `No` are dropped after this transformation.

---

## Setup & Requirements

### Prerequisites

- Python 3.7+
- Jupyter Notebook or Google Colab
- The following Python libraries:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - openpyxl (for reading Excel files)

### Installation

Install the required libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn openpyxl
```

---

## How to Run

1. **Download the files**: Ensure you have `Source_code_(SSANAH_Group).ipynb` and `Real estate.xlsx` in the same directory.
2. **Open the notebook**: Launch Jupyter Notebook and open `Source_code_(SSANAH_Group).ipynb`.
3. **Run all cells**: Execute the notebook cells sequentially. The notebook will:
   - Load and preprocess the data
   - Perform exploratory data analysis (EDA)
   - Engineer features
   - Train and evaluate multiple regression models
   - Visualize results

**Note:** The notebook will also save a processed version of the dataset as `Riyalstate.xlsx`.

---

## Analysis & Modeling Steps

### 1. Data Loading & Preprocessing
- Load the dataset from Excel.
- Feature engineering: Compute `X5 Vector Mag` and drop unnecessary columns.
- Save the processed dataset.

### 2. Exploratory Data Analysis (EDA)
- Generate scatterplots for all features vs. the target.
- Plot a heatmap to visualize feature correlations.

### 3. Data Preparation
- Split the data into features (`X`) and target (`y`).
- Standardize features using `StandardScaler`.
- Split into training and testing sets.

### 4. Regression Modeling

#### a. **Linear Regression**
- Fit a linear regression model.
- Evaluate using R², MAE, MSE, and RMSE.
- Visualize predictions and residuals.

#### b. **Random Forest Regression**
- Fit a Random Forest regressor.
- Evaluate and visualize as above.

#### c. **PCA + Random Forest Pipeline**
- Create a pipeline with scaling, PCA (retain 95% variance), and Random Forest.
- Fit and evaluate the pipeline.

#### d. **Polynomial Regression**
- Use `PolynomialFeatures` (default degree=1, can be changed).
- Fit a linear regression on polynomial features.
- Evaluate and visualize.

---

## Results & Evaluation

For each model, the following metrics are reported:
- **R² (Coefficient of Determination)**
- **MAE (Mean Absolute Error)**
- **MSE (Mean Squared Error)**
- **RMSE (Root Mean Squared Error)**

Visualizations include:
- Scatterplots of actual vs. predicted values
- Residual plots

**Note:** The Random Forest model typically achieves the best performance, as shown by higher R² and lower error metrics.

---

## Outputs

- **Riyalstate.xlsx**: Processed dataset with engineered features.
- **Plots**: Visualizations of model performance and residuals (displayed in the notebook).
- **Metrics**: Printed in the notebook output for each model.

---

## References

- [scikit-learn documentation](https://scikit-learn.org/stable/documentation.html)
- [Pandas documentation](https://pandas.pydata.org/)
- [Seaborn documentation](https://seaborn.pydata.org/)
- [Original dataset source](https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set)


---

## License

This project is for educational purposes.

---

**For any questions or issues, please contact the project authors or your course instructor.** 