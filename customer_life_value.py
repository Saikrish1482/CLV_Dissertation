#!/usr/bin/env python3
"""
Customer Life Value (CLV) Prediction Analysis

This script performs comprehensive Customer Life Value analysis using various machine learning algorithms.
It includes data loading, preprocessing, feature engineering, and model comparison.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as custlifepn
import matplotlib.pyplot as clvmatplt
import math as clvMath

# Scikit-learn imports
from sklearn.model_selection import train_test_split as clvtrtst
from sklearn.metrics import r2_score as clvR2Scr
from sklearn.metrics import mean_absolute_error as clvMAE
from sklearn.metrics import mean_squared_error as clvMSE
from sklearn.model_selection import GridSearchCV as clvGrdSr

# Model imports
from sklearn.tree import DecisionTreeRegressor as clvDecisionTreeReg
from sklearn.linear_model import PassiveAggressiveRegressor as clvPassiveReg
from sklearn.ensemble import RandomForestRegressor as clvRandomForestReg
from sklearn.ensemble import GradientBoostingRegressor as clvGradientBoostReg
from sklearn.ensemble import VotingRegressor as clVotingReg

# =============================================================================
# DATA LOADING
# =============================================================================
print("Loading data...")
life1 = custlifepn.read_csv("Returns.csv")
life2 = custlifepn.read_csv("Suppliers.csv")
life3 = custlifepn.read_csv("Customers.csv")
life4 = custlifepn.read_csv("Orders.csv")
life5 = custlifepn.read_csv("Payment_info.csv")
life6 = custlifepn.read_csv("Products.csv")

# Display data shapes
print("Returns Data Shape      = ", life1.shape)
print("Suppliers Data Shape    = ", life2.shape)
print("Customers Data Shape    = ", life3.shape)
print("Orders Data Shape       = ", life4.shape)
print("Payment Info Data Shape = ", life5.shape)
print("Products Data Shape     = ", life6.shape)

# =============================================================================
# DATA EXPLORATION
# =============================================================================
print("\nData exploration:")
print("Returns Data - First 5 rows:")
print(life1.head())

print("\nSuppliers Data - First 5 rows:")
print(life2.head())

print("\nCustomers Data - First 5 rows:")
print(life3.head())

print("\nOrders Data - Last 5 rows:")
print(life4.tail())

print("\nPayment Info Data - Last 5 rows:")
print(life5.tail())

print("\nProducts Data - Last 5 rows:")
print(life6.tail())

# =============================================================================
# SAMPLE DATA EXPORT (First 1000 rows)
# =============================================================================
print("\nCreating sample datasets...")
sample_life1 = life1[0:1000]
file_path = r"sample_life1.csv"
sample_life1.to_csv(file_path, index=False)

sample_life2 = life2[0:1000]
file_path = r"sample_life2.csv"
sample_life2.to_csv(file_path, index=False)

sample_life3 = life3[0:1000]
file_path = r"sample_life3.csv"
sample_life3.to_csv(file_path, index=False)

sample_life4 = life4[0:1000]
file_path = r"sample_life4.csv"
sample_life4.to_csv(file_path, index=False)

sample_life5 = life5[0:1000]
file_path = r"sample_life5.csv"
sample_life5.to_csv(file_path, index=False)

sample_life6 = life6[0:1000]
file_path = r"sample_life6.csv"
sample_life6.to_csv(file_path, index=False)

print("Sample datasets created successfully!")

# =============================================================================
# DATA MERGING
# =============================================================================
print("\nMerging datasets...")

## Merging Data Frame 1
merge1 = custlifepn.merge(life3, life4, on='customer_id', suffixes=('_df1', '_df2'))
print("Merged DataFrame1 with suffixes:")
merge1.info()

## Merging Data Frame 2
merge2 = custlifepn.merge(merge1, life5, on="order_id", suffixes=('_df1', '_df2'))
print("Merged DataFrame2 with suffixes:")
merge2.info()

## Merging Data Frame 3
merge3 = custlifepn.merge(merge2, life6, on="product_id", suffixes=('_df1', '_df2'))
print("Merged DataFrame3 with suffixes:")
merge3.info()

# =============================================================================
# CLV CALCULATION
# =============================================================================
print("\nCalculating Customer Lifetime Value components...")

## Calculate total revenue per customer
customer_revenue = merge3.groupby('customer_id')['total_price'].sum().reset_index()
customer_revenue.columns = ['customer_id', 'total_revenue']
print("Customer revenue calculated")

## Calculate purchase frequency per customer
purchase_frequency = merge3.groupby('customer_id')['order_id'].count().reset_index()
purchase_frequency.columns = ['customer_id', 'purchase_frequency']
print("Purchase frequency calculated")

## Calculate average order value per customer
merge3['average_order_value'] = merge3['total_price'] / merge3['quantity']
avg_order_value = merge3.groupby('customer_id')['average_order_value'].mean().reset_index()
avg_order_value.columns = ['customer_id', 'average_order_value']
print("Average order value calculated")

## Convert date columns to datetime
merge3['order_date'] = custlifepn.to_datetime(merge3['order_date'])
merge3['created_at'] = custlifepn.to_datetime(merge3['created_at'])

## Calculate customer lifespan in years
merge3['customer_lifespan'] = (merge3['order_date'].max() - merge3.groupby('customer_id')['order_date'].transform('min')).dt.days / 365
lifespan = merge3.groupby('customer_id')['customer_lifespan'].max().reset_index()
lifespan.columns = ['customer_id', 'customer_lifespan']
print("Customer lifespan calculated")

# Merge all metrics
clv_data = customer_revenue.merge(purchase_frequency, on='customer_id')
clv_data = clv_data.merge(avg_order_value, on='customer_id')
clv_data = clv_data.merge(lifespan, on='customer_id')

# Calculate CLV
clv_data['CLV'] = clv_data['average_order_value'] * clv_data['purchase_frequency'] * clv_data['customer_lifespan']

print("Customer Lifetime Value (CLV) data:")
print(clv_data)

# =============================================================================
# DATA QUALITY CHECK
# =============================================================================
print("\nData quality check:")
print("Null in Customer Life Value Prediction:", clv_data.isnull().values.sum())
print("Duplicates in Customer Life Value Prediction:", clv_data.duplicated().values.sum())

# =============================================================================
# DATA VISUALIZATION
# =============================================================================
print("\nCreating CLV histogram...")
clvmatplt.hist(clv_data['CLV'], bins=6, color='cyan')
clvmatplt.title('Histogram for CLV')
clvmatplt.show()

# =============================================================================
# MODEL PREPARATION
# =============================================================================
print("\nPreparing data for modeling...")
clv_X = clv_data.drop('CLV', axis=1)
clv_Y = clv_data['CLV']

# Train-test split
clvX_tr, clvX_ts, clvY_tr, clvY_ts = clvtrtst(clv_X, clv_Y, test_size=0.2, random_state=12)
print("Training CLV :", clvX_tr.shape)
print("Testing CLV  :", clvX_ts.shape)

# =============================================================================
# DECISION TREE REGRESSOR
# =============================================================================
print("\n" + "="*50)
print("DECISION TREE REGRESSOR")
print("="*50)

# Grid search for best parameters
clvGridParameters = {
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'splitter': ['best', 'random'],
    'max_depth': [5, 10, 2, 3]
}

clvMdl = clvDecisionTreeReg()
clvMdl = clvGrdSr(clvMdl, clvGridParameters, cv=2)
clvMdl.fit(clvX_tr.sample(950, random_state=12), clvY_tr.sample(950, random_state=12))
print("Selected Parameters for Decision Tree Regressor:")
print(clvMdl.best_params_)

# Training
clvMdl = clvDecisionTreeReg(**clvMdl.best_params_)
clvMdl.fit(clvX_tr, clvY_tr)

# Testing
clvYpred = clvMdl.predict(clvX_ts)
print("\nDecision Tree in Customer Life Value Prediction")
print("R2 Score :", clvR2Scr(clvY_ts, clvYpred)*100)
print("MAE      :", clvMAE(clvY_ts, clvYpred))
print("MSE      :", clvMSE(clvY_ts, clvYpred))
print("RMSE     :", clvMath.sqrt(clvMSE(clvY_ts, clvYpred)))

# =============================================================================
# PASSIVE AGGRESSIVE REGRESSOR
# =============================================================================
print("\n" + "="*50)
print("PASSIVE AGGRESSIVE REGRESSOR")
print("="*50)

# Grid search for best parameters
clvGridParameters = {
    'C': [1.0, 0.001, 0.05, 0.003],
    'fit_intercept': [True, False],
    'max_iter': [500, 300, 200, 100]
}

clvMdl = clvPassiveReg()
clvMdl = clvGrdSr(clvMdl, clvGridParameters, cv=2)
clvMdl.fit(clvX_tr.sample(950, random_state=12), clvY_tr.sample(950, random_state=12))
print("Selected Parameters for Passive Aggressive Regressor:")
print(clvMdl.best_params_)

# Training
clvMdl = clvPassiveReg(**clvMdl.best_params_)
clvMdl.fit(clvX_tr, clvY_tr)

# Testing
clvYpred = clvMdl.predict(clvX_ts)
print("\nPassive Aggressive in Customer Life Value Prediction")
print("R2 Score :", clvR2Scr(clvY_ts, clvYpred)*100)
print("MAE      :", clvMAE(clvY_ts, clvYpred))
print("MSE      :", clvMSE(clvY_ts, clvYpred))
print("RMSE     :", clvMath.sqrt(clvMSE(clvY_ts, clvYpred)))

# =============================================================================
# RANDOM FOREST REGRESSOR
# =============================================================================
print("\n" + "="*50)
print("RANDOM FOREST REGRESSOR")
print("="*50)

# Grid search for best parameters
clvGridParameters = {
    'n_estimators': [100, 200, 400],
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'max_depth': [5, 10, 2, 3]
}

clvMdl = clvRandomForestReg()
clvMdl = clvGrdSr(clvMdl, clvGridParameters, cv=2)
clvMdl.fit(clvX_tr.sample(950, random_state=12), clvY_tr.sample(950, random_state=12))
print("Selected Parameters for Random Forest Regressor:")
print(clvMdl.best_params_)

# Training
clvMdl = clvRandomForestReg(**clvMdl.best_params_)
clvMdl.fit(clvX_tr, clvY_tr)

# Testing
clvYpred = clvMdl.predict(clvX_ts)
print("\nRandom Forest in Customer Life Value Prediction")
print("R2 Score :", clvR2Scr(clvY_ts, clvYpred)*100)
print("MAE      :", clvMAE(clvY_ts, clvYpred))
print("MSE      :", clvMSE(clvY_ts, clvYpred))
print("RMSE     :", clvMath.sqrt(clvMSE(clvY_ts, clvYpred)))

# =============================================================================
# GRADIENT BOOSTING REGRESSOR
# =============================================================================
print("\n" + "="*50)
print("GRADIENT BOOSTING REGRESSOR")
print("="*50)

# Grid search for best parameters
clvGridParameters = {
    'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
    'learning_rate': [0.1, 0.5, 0.001, 1.0],
    'n_estimators': [100, 200, 400]
}

clvMdl = clvGradientBoostReg()
clvMdl = clvGrdSr(clvMdl, clvGridParameters, cv=2)
clvMdl.fit(clvX_tr.sample(950, random_state=12), clvY_tr.sample(950, random_state=12))
print("Selected Parameters for Gradient Boosting Regressor:")
print(clvMdl.best_params_)

# Training
clvMdl = clvGradientBoostReg(**clvMdl.best_params_)
clvMdl.fit(clvX_tr, clvY_tr)

# Testing
clvYpred = clvMdl.predict(clvX_ts)
print("\nGradient Boosting in Customer Life Value Prediction")
print("R2 Score :", clvR2Scr(clvY_ts, clvYpred)*100)
print("MAE      :", clvMAE(clvY_ts, clvYpred))
print("MSE      :", clvMSE(clvY_ts, clvYpred))
print("RMSE     :", clvMath.sqrt(clvMSE(clvY_ts, clvYpred)))

# =============================================================================
# ENSEMBLE METHODS - COMBINED ALGORITHMS
# =============================================================================
print("\n" + "="*70)
print("ENSEMBLE METHODS - COMBINED ALGORITHMS")
print("="*70)

# =============================================================================
# Decision Tree and Passive Aggressive Regressor Combination
# =============================================================================
print("\nDecision Tree and Passive Aggressive Regressor Combination")
print("-" * 60)

clvDec = clvDecisionTreeReg(criterion='poisson', max_depth=10, splitter='random')
clvPass = clvPassiveReg(C=0.003, fit_intercept=True, max_iter=500)

clvMdl = clVotingReg(estimators=[('DTR', clvDec), ('PAR', clvPass)])
clvMdl.fit(clvX_tr, clvY_tr)
print(clvMdl)

# Testing
clvYpred = clvMdl.predict(clvX_ts)
print("\nCombination of Decision Tree and Passive Aggressive Regressor")
print("R2 Score :", clvR2Scr(clvY_ts, clvYpred)*100)
print("MAE      :", clvMAE(clvY_ts, clvYpred))
print("MSE      :", clvMSE(clvY_ts, clvYpred))
print("RMSE     :", clvMath.sqrt(clvMSE(clvY_ts, clvYpred)))

# =============================================================================
# Passive Aggressive and Random Forest Regressor Combination
# =============================================================================
print("\nPassive Aggressive and Random Forest Regressor Combination")
print("-" * 60)

clvPass = clvPassiveReg(C=0.003, fit_intercept=True, max_iter=500)
clvRand = clvRandomForestReg(criterion='poisson', max_depth=10, n_estimators=400)
clvMdl = clVotingReg(estimators=[('PAR', clvPass), ('RFR', clvRand)])
clvMdl.fit(clvX_tr, clvY_tr)
print(clvMdl)

# Testing
clvYpred = clvMdl.predict(clvX_ts)
print("\nCombination of Passive Aggressive and Random Forest")
print("R2 Score :", clvR2Scr(clvY_ts, clvYpred)*100)
print("MAE      :", clvMAE(clvY_ts, clvYpred))
print("MSE      :", clvMSE(clvY_ts, clvYpred))
print("RMSE     :", clvMath.sqrt(clvMSE(clvY_ts, clvYpred)))

# =============================================================================
# Random Forest and Gradient Boosting Regressor Combination
# =============================================================================
print("\nRandom Forest and Gradient Boosting Regressor Combination")
print("-" * 60)

clvRand = clvRandomForestReg(criterion='poisson', max_depth=10, n_estimators=400)
clvGrad = clvGradientBoostReg(learning_rate=0.1, loss='squared_error', n_estimators=400)

clvMdl = clVotingReg(estimators=[('RFR', clvRand), ('GBR', clvGrad)])
clvMdl.fit(clvX_tr, clvY_tr)
print(clvMdl)

# Testing
clvYpred = clvMdl.predict(clvX_ts)
print("\nCombination of Random Forest and Gradient Boosting Regressor")
print("R2 Score :", clvR2Scr(clvY_ts, clvYpred)*100)
print("MAE      :", clvMAE(clvY_ts, clvYpred))
print("MSE      :", clvMSE(clvY_ts, clvYpred))
print("RMSE     :", clvMath.sqrt(clvMSE(clvY_ts, clvYpred)))

# =============================================================================
# Gradient Boosting and Decision Tree Regressor Combination
# =============================================================================
print("\nGradient Boosting and Decision Tree Regressor Combination")
print("-" * 60)

clvGrad = clvGradientBoostReg(learning_rate=0.1, loss='squared_error', n_estimators=400)
clvDec = clvDecisionTreeReg(criterion='poisson', max_depth=10, splitter='random')

clvMdl = clVotingReg(estimators=[('GBR', clvGrad), ('DTR', clvDec)])
clvMdl.fit(clvX_tr, clvY_tr)
print(clvMdl)

# Testing
clvYpred = clvMdl.predict(clvX_ts)
print("\nCombination of Gradient Boosting and Decision Tree Regressor")
print("R2 Score :", clvR2Scr(clvY_ts, clvYpred)*100)
print("MAE      :", clvMAE(clvY_ts, clvYpred))
print("MSE      :", clvMSE(clvY_ts, clvYpred))
print("RMSE     :", clvMath.sqrt(clvMSE(clvY_ts, clvYpred)))

# =============================================================================
# All Four Algorithms Combined
# =============================================================================
print("\nDecision Tree, Passive Aggressive, Random Forest and Gradient Boosting Regressor Combination")
print("-" * 90)

clvDec = clvDecisionTreeReg(criterion='poisson', max_depth=10, splitter='random')
clvPass = clvPassiveReg(C=0.003, fit_intercept=True, max_iter=500)
clvRand = clvRandomForestReg(criterion='poisson', max_depth=10, n_estimators=400)
clvGrad = clvGradientBoostReg(learning_rate=0.1, loss='squared_error', n_estimators=400)

clvMdl = clVotingReg(estimators=[('DTR', clvDec), ('PAR', clvPass), ('RFR', clvRand), ('GBR', clvGrad)])
clvMdl.fit(clvX_tr, clvY_tr)
print(clvMdl)

# Testing
clvYpred = clvMdl.predict(clvX_ts)
print("\nCombination of Decision Tree, Passive Aggressive, Random Forest and Gradient Boosting Regressors")
print("R2 Score :", clvR2Scr(clvY_ts, clvYpred)*100)
print("MAE      :", clvMAE(clvY_ts, clvYpred))
print("MSE      :", clvMSE(clvY_ts, clvYpred))
print("RMSE     :", clvMath.sqrt(clvMSE(clvY_ts, clvYpred)))

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)