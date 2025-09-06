# Bank Term Deposit Prediction

A comprehensive machine learning project to predict customer subscription to term deposits using bank marketing campaign data. This project implements a modular, scalable approach to binary classification with multiple algorithms and thorough evaluation.

## Project Overview

This project analyzes bank marketing campaign data to predict whether clients will subscribe to term deposits. The solution provides actionable insights for optimizing marketing strategies and improving campaign ROI.

### Business Problem
Banks invest significant resources in marketing campaigns to promote term deposits. By accurately predicting which clients are most likely to subscribe, banks can:
- Optimize marketing spend by targeting high-probability prospects
- Increase campaign conversion rates
- Improve customer experience by reducing unwanted marketing contact
- Maximize return on investment for marketing activities

### Key Features
- **Modular Architecture**: Clean, reusable code structure for easy maintenance and extension
- **Comprehensive EDA**: In-depth exploratory data analysis with visualizations
- **Advanced Feature Engineering**: Creates meaningful features and handles multicollinearity
- **Multiple Algorithms**: Implements and compares Naive Bayes and Decision Tree models
- **Thorough Evaluation**: Complete model assessment with metrics, visualizations, and reports
- **Automated Pipeline**: End-to-end execution from raw data to production-ready models

## Dataset Information

The dataset contains information about bank marketing campaigns with the following characteristics:

### Target Variable
- **y**: Client subscription to term deposit (binary: 'yes'/'no')

### Features
- **Demographic**: Age, job, marital status, education, default status
- **Financial**: Housing loan, personal loan status
- **Campaign**: Contact method, month, day of week, duration, number of contacts
- **Economic**: Employment variation rate, consumer price index, confidence index

### Data Quality
- **Shape**: ~41,000 rows × 20+ features
- **Missing Values**: Minimal missing data
- **Class Distribution**: Imbalanced dataset (majority class: no subscription)

## Project Structure

```
bank-term-deposit-prediction/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration and constants
│   ├── data_processing.py        # Data loading, cleaning, EDA
│   ├── feature_engineering.py    # Feature transformations
│   ├── model_training.py         # Model definitions and training
│   ├── evaluation.py             # Model evaluation and metrics
│   └── main.py                   # Pipeline orchestrator
├── data/
│   └── bank_data.csv            # Dataset file
├── models/                      # Saved model files (.pkl)
├── results/                     # Output files, plots, reports
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Installation & Setup

1. **Clone the repository and navigate to project directory**
2. **Create virtual environment:**
   ```bash
   python -m venv bank_env
   source bank_env/bin/activate  # On Windows: bank_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data:**
   - Place `bank_data.csv` in the `data/` directory
   - Ensure the file uses semicolon (;) as delimiter

## Usage

### Complete Pipeline
Run the entire machine learning pipeline:
```python
python src/main.py
```

### Individual Components
Run specific pipeline steps:
```python
from src.data_processing import main_data_processing
from src.feature_engineering import main_feature_engineering
from src.model_training import main_model_training
from src.evaluation import main_evaluation

# Load and explore data
df, summary = main_data_processing()

# Engineer features
processed_data = main_feature_engineering(df)

# Train models
training_results = main_model_training(processed_data)

# Evaluate models
evaluation_results = main_evaluation(
    training_results['training_results'],
    processed_data['X_test'],
    processed_data['y_test']
)
```

## Key Insights & Findings

### Data Insights
- **Class Imbalance**: ~88% customers do not subscribe to term deposits
- **High Correlation**: Strong correlation between economic indicators (emp.var.rate, euribor3m, nr.employed)
- **Seasonal Patterns**: March and December show higher subscription rates
- **Contact Method**: Cellular contact generally outperforms telephone

### Model Performance
The project implements and compares three algorithms:

1. **Naive Bayes**
   - Fast training and prediction
   - Good baseline performance
   - Handles categorical features well

2. **Decision Tree (Gini)**
   - Interpretable decision rules
   - Handles mixed data types
   - Provides feature importance

3. **Decision Tree (Entropy)**
   - Alternative splitting criterion
   - Comparable performance to Gini
   - Different decision boundaries

### Business Recommendations
1. **Target Segments**: Focus campaigns on students and retired individuals
2. **Timing**: Schedule major campaigns in March and December
3. **Contact Method**: Prioritize cellular over telephone contact
4. **Previous Success**: Heavily weight previous campaign success in targeting decisions

## Technical Highlights

### Feature Engineering
- **pdays999 Feature**: Binary indicator for customers with no previous contact
- **Multicollinearity Handling**: Removes highly correlated economic indicators
- **Categorical Encoding**: One-hot encoding with drop_first=True
- **Data Validation**: Comprehensive missing value and data quality checks

### Model Evaluation
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, AUC
- **Visual Analysis**: Confusion matrices, ROC curves, model comparison plots
- **Statistical Reports**: Detailed classification reports for each model
- **Model Persistence**: Automated saving of trained models with timestamps

### Code Quality
- **Modular Design**: Separates concerns across logical modules
- **Error Handling**: Robust exception handling throughout pipeline
- **Logging**: Comprehensive logging for debugging and monitoring
- **Documentation**: Detailed docstrings and inline comments

## Results & Outputs

The pipeline generates several outputs:

### Models
- **Saved Models**: Serialized model files in `models/` directory
- **Training Results**: Comprehensive training metrics and metadata

### Visualizations
- **Target Distribution**: Class balance visualization
- **Subscription Rates**: Analysis by categorical variables
- **Correlation Matrix**: Feature correlation heatmaps
- **Confusion Matrices**: Model performance visualization
- **ROC Curves**: Model discrimination analysis

### Reports
- **Model Comparison**: Side-by-side performance metrics
- **Feature Importance**: Key predictive features (for tree models)
- **Evaluation Summary**: Comprehensive model assessment

## Future Enhancements

### Model Improvements
- **Advanced Algorithms**: Random Forest, Gradient Boosting, Neural Networks
- **Hyperparameter Tuning**: Grid search and cross-validation
- **Ensemble Methods**: Model stacking and voting classifiers
- **Class Balancing**: SMOTE, class weights, or cost-sensitive learning

### Feature Engineering
- **Derived Features**: Interaction terms, polynomial features
- **Time-based Features**: Seasonality, trends, cyclic patterns
- **External Data**: Economic indicators, market conditions
- **Text Mining**: Analysis of campaign notes or customer feedback

### Production Readiness
- **API Endpoint**: REST API for real-time predictions
- **Model Monitoring**: Performance tracking and drift detection
- **A/B Testing**: Framework for testing model improvements
- **Scalability**: Distributed processing for large datasets

## Dependencies

Key libraries used in this project:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and metrics
- **matplotlib/seaborn**: Data visualization
- **joblib**: Model persistence

See `requirements.txt` for complete dependency list with versions.

## Contributing

This project follows professional data science practices:
1. **Code Quality**: PEP 8 style guidelines
2. **Testing**: Unit tests for core functions
3. **Documentation**: Comprehensive docstrings
4. **Version Control**: Meaningful commit messages

## License

This project is designed for educational and professional portfolio purposes.

---

**Contact**: For questions about this project or collaboration opportunities, please reach out through the project repository.