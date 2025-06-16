# Cervical Cancer Prediction Using XGBoost

This project applies machine learning techniques to predict cervical cancer risk using the UCI Cervical Cancer dataset. It utilizes the XGBoost classifier to identify patterns in patient data and estimate the likelihood of cervical cancer.

##  Files

- `Cervical Cancer Prediction Using XG-boost algorithm.ipynb`: The main Jupyter Notebook containing data preprocessing, exploratory data analysis, model training, and evaluation.
- `cervical_cancer.csv`: The dataset used for training and testing, originally sourced from the UCI Machine Learning Repository.

##  Dataset

The dataset includes demographic and medical risk factors for cervical cancer. Some features include:
- Age  
- Number of sexual partners  
- Smoking habits  
- STDs history  
- HPV status  

Missing values are handled and categorical values are appropriately encoded.

##  Model

- **Algorithm**: XGBoost Classifier  
- **Hyperparameters**: `learning_rate=0.1`, `max_depth=5`, `n_estimators=10`  
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix  

##  Key Steps

1. Load and preprocess the dataset
2. Handle missing values (`?` replaced with `NaN`)
3. Visualize data with heatmaps and distributions
4. Train the XGBoost model
5. Evaluate model performance

##  Results

The model performs well on the dataset and offers insights into which factors are most predictive of cervical cancer risk.

##  Getting Started

1. Clone the repo
2. Open the notebook `Cervical Cancer Prediction Using XG-boost algorithm.ipynb` in Jupyter or Google Colab
3. Run all cells to view the results

```bash
pip install pandas numpy seaborn scikit-learn xgboost
