# Customer Life Value (CLV) Prediction Analysis

A comprehensive machine learning project for predicting Customer Life Value using various regression algorithms and ensemble methods.

## 📊 Project Overview

This project analyzes customer data to predict Customer Life Value (CLV) using multiple machine learning approaches. The analysis includes data preprocessing, feature engineering, model comparison, and ensemble methods to find the best performing algorithm for CLV prediction.

## 🎯 Objectives

- **Data Integration**: Merge multiple datasets to create a comprehensive customer view
- **Feature Engineering**: Calculate CLV components (revenue, frequency, order value, lifespan)
- **Model Comparison**: Evaluate multiple regression algorithms
- **Ensemble Learning**: Test combinations of algorithms for improved performance
- **Performance Analysis**: Compare models using multiple evaluation metrics

## 📁 Dataset Description

The project uses 6 CSV files containing different aspects of customer and business data:

| File | Description |
|------|-------------|
| `Returns.csv` | Product return information |
| `Suppliers.csv` | Supplier details |
| `Customers.csv` | Customer demographics and information |
| `Orders.csv` | Order transaction details |
| `Payment_info.csv` | Payment and transaction information |
| `Products.csv` | Product catalog and details |

## 🔧 Technical Requirements

### Dependencies
```bash
pip install pandas
pip install matplotlib
pip install scikit-learn
```

### Python Version
- Python 3.7 or higher

## 🚀 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/customer-life-value-prediction.git
   cd customer-life-value-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**
   - Place all 6 CSV files in the same directory as the script
   - Ensure file names match: `Returns.csv`, `Suppliers.csv`, `Customers.csv`, `Orders.csv`, `Payment_info.csv`, `Products.csv`

4. **Run the analysis**
   ```bash
   python customer_life_value.py
   ```

## 📈 Features

### Data Processing
- **Data Loading**: Automated loading of multiple CSV files
- **Data Exploration**: Shape analysis and sample data display
- **Data Merging**: Sequential joining of datasets on key columns
- **Quality Checks**: Null value and duplicate detection

### CLV Calculation
The script calculates CLV using the formula:
```
CLV = Average Order Value × Purchase Frequency × Customer Lifespan
```

Components calculated:
- **Total Revenue**: Sum of all purchases per customer
- **Purchase Frequency**: Number of orders per customer
- **Average Order Value**: Mean transaction value per customer
- **Customer Lifespan**: Time between first and last purchase (in years)

### Machine Learning Models

#### Individual Algorithms
1. **Decision Tree Regressor**
   - Grid search optimization
   - Parameters: criterion, splitter, max_depth

2. **Passive Aggressive Regressor**
   - Grid search optimization  
   - Parameters: C, fit_intercept, max_iter

3. **Random Forest Regressor**
   - Grid search optimization
   - Parameters: n_estimators, criterion, max_depth

4. **Gradient Boosting Regressor**
   - Grid search optimization
   - Parameters: loss, learning_rate, n_estimators

#### Ensemble Methods
- Decision Tree + Passive Aggressive
- Passive Aggressive + Random Forest  
- Random Forest + Gradient Boosting
- Gradient Boosting + Decision Tree
- All Four Algorithms Combined

### Evaluation Metrics
- **R² Score**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error  
- **RMSE**: Root Mean Squared Error

## 📊 Output

The script provides:
- Data shape information for all datasets
- Sample data previews
- Merged dataset statistics
- CLV distribution histogram
- Model performance comparison
- Best parameters for each algorithm
- Detailed evaluation metrics for all models

## 🔍 Results Analysis

### Model Performance Comparison
The script tests multiple algorithms and their combinations to identify:
- Best individual performer
- Most effective ensemble combination
- Optimal hyperparameters for each model
- Performance trade-offs between different approaches

### Sample Output Format
```
Decision Tree in Customer Life Value Prediction
R2 Score : 85.23
MAE      : 156.78
MSE      : 45234.56
RMSE     : 212.68
```

## 📁 Project Structure
```
customer-life-value-prediction/
│
├── customer_life_value.py    # Main analysis script
├── README.md                 # This file
├── requirements.txt          # Python dependencies
│
├── data/                     # Data directory (create this)
│   ├── Returns.csv
│   ├── Suppliers.csv  
│   ├── Customers.csv
│   ├── Orders.csv
│   ├── Payment_info.csv
│   └── Products.csv
│
└── output/                   # Generated sample files
    ├── sample_life1.csv
    ├── sample_life2.csv
    ├── sample_life3.csv
    ├── sample_life4.csv
    ├── sample_life5.csv
    └── sample_life6.csv
```

## 🎨 Visualization

The project includes:
- **CLV Distribution Histogram**: Visual representation of customer lifetime values
- **Model Performance Comparison**: Evaluation metrics across different algorithms

## 🔄 Workflow

1. **Data Loading** → Load all CSV files
2. **Data Exploration** → Examine data structure and quality
3. **Data Integration** → Merge datasets on key relationships
4. **Feature Engineering** → Calculate CLV components
5. **Data Preparation** → Train-test split and preprocessing
6. **Model Training** → Grid search and algorithm training
7. **Model Evaluation** → Performance assessment and comparison
8. **Ensemble Testing** → Combination algorithm evaluation

## 🎯 Business Applications

- **Customer Segmentation**: Identify high-value customers
- **Marketing Strategy**: Target customers based on predicted CLV
- **Resource Allocation**: Focus efforts on customers with highest potential value
- **Retention Programs**: Develop strategies for different CLV segments
- **Revenue Forecasting**: Predict future customer value

## 📝 Notes

- The script creates sample datasets (first 1000 rows) for testing purposes
- All models use cross-validation for robust parameter selection
- Ensemble methods combine predictions from multiple algorithms
- Performance metrics help identify the most suitable model for your specific use case

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

Project Link: [https://github.com/yourusername/customer-life-value-prediction](https://github.com/yourusername/customer-life-value-prediction)

## 🙏 Acknowledgments

- Scikit-learn documentation and community
- Pandas and Matplotlib libraries
- Machine learning best practices and methodologies

---

### 📊 Quick Start Example

```python
# Quick example of running individual components
import pandas as pd
from customer_life_value import *

# Load your data
life3 = pd.read_csv("Customers.csv")
life4 = pd.read_csv("Orders.csv")

# The script handles the rest automatically!
```

**Happy Analyzing! 🚀**
