# Used Car Sales Analysis

Data analytics project from my 5th semester of Computer Science and Engineering at PES University (2021), done with Vedant Mantri and Pranav Mekal Mahesh.

The goal was to study the online used car market and compare different models for predicting used car prices from listing attributes.

## What is in this repo

| File | Description |
| --- | --- |
| `Used Cars_DA.ipynb` | Notebook with data cleaning, exploratory analysis and modelling |
| `vehicles.csv` | Kaggle dataset of Craigslist used vehicle listings (about 1.4 GB, stored with Git LFS) |
| `5_AlphaQ_FinalReport.pdf` | Final report written as a short paper |
| `5_AlphaQ_LiteratureReview.pdf` | Literature review |

## Approach

1. Cleaned and preprocessed the data (missing values, outliers and encoding of categorical columns)
2. Explored the data with visualisations
3. Split the data with `StratifiedKFold`
4. Trained Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM and MLP regressors, with and without hyperparameter tuning
5. Compared the models using R² and RMSE on the test data

## Running it

The notebook was written in Google Colab. To run it locally install Git LFS so that `vehicles.csv` downloads properly and then update the dataset path in the first cells.

```bash
git lfs install
git clone https://github.com/Mayuravarsha/Data-Analytics.git
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
```
