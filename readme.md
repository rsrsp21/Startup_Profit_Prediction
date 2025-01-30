# Startup Profit Prediction

## Overview

This project utilizes multiple machine learning models to predict a startup's profit based on its expenses. The dataset used for training includes R&D Spend, Administration Cost, and Marketing Spend as independent variables and Profit as the dependent variable.

## Files in Repository

- `Data Science Profit Prediction.ipynb`: Jupyter Notebook containing the code for data analysis, model training, and evaluation.
- `50_Startups.csv`: Dataset containing information about startups and their expenses.

## Technologies Used

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- XGBoost
- TensorFlow

## Models Implemented

The following regression models were used for prediction:

- Linear Regression
- Lasso Regression
- Ridge Regression
- ElasticNet Regression
- Random Forest Regressor
- Decision Tree Regressor
- Support Vector Regressor (SVR)
- XGBoost Regressor
- Feedforward Neural Network (FNN)

## Model Evaluation

Models were evaluated using the following metrics:

- **R2 Score**
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**

The models were ranked based on these metrics, and the best-performing model was determined.

## Best Performing Model

Based on ranking using R2 Score, MAE, and MSE, the **Random Forest Regressor** was found to be the best model.

## Usage

Run the Jupyter Notebook (`startup_profit_prediction.ipynb`) to:

1. Load and preprocess the dataset.
2. Train multiple regression models.
3. Evaluate model performance.
4. Identify the best model.
5. Predict the profit of a startup based on user inputs.

To predict profit using the trained **Random Forest Regressor**, enter values for R&D Spend, Administration Cost, and Marketing Spend when prompted.

## Installation

1. Clone the repository:
   ```sh
   git clone <repository-url>
   ```
2. Install dependencies:
   ```sh
   pip install pandas numpy seaborn matplotlib scikit-learn xgboost tensorflow
   ```
3. Open the Jupyter Notebook and run the cells step by step.

## Example Prediction

```
Enter R & D Spend : 150000
Enter Administration Cost : 120000
Enter Marketing Spend : 300000
The predicted profit of the Startup is : 192631.1
```

## License

This project is for educational purposes and is open-source.

## Author

Sri Ram Sai Pavan Relangi\
Contact: [rsrsp21@gmail.com](mailto\:rsrsp21@gmail.com)

**This project was completed as part of an internship at Exposys Data Labs.**
