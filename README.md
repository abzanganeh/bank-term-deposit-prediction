# Bank Term Deposit Prediction

A comprehensive machine learning project to predict customer subscription to term deposits using bank marketing campaign data. This project implements a modular, scalable approach to binary classification with multiple algorithms and thorough evaluation.

## Project Overview

This project analyzes bank marketing campaign data to predict whether clients will subscribe to term deposits. The solution provides actionable insights for optimizing marketing strategies and improving campaign ROI.

The project includes both a basic pipeline (for learning and quick experimentation) and an advanced pipeline with state-of-the-art machine learning techniques including hyperparameter tuning, threshold optimization, and model stacking.

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
- **Multiple Algorithms**: Implements and compares 8+ models including advanced algorithms
- **Hyperparameter Tuning**: Automated optimization using Optuna for best performance
- **Threshold Optimization**: Dynamic threshold tuning for improved predictions
- **Model Stacking**: Ensemble methods combining multiple models for superior performance
- **Advanced Evaluation**: Comprehensive metrics, ROC curves, calibration analysis
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
│   ├── model_training.py         # Basic model definitions and training
│   ├── evaluation.py             # Basic model evaluation and metrics
│   ├── enhanced_model_training.py # Advanced models, tuning, stacking
│   ├── enhanced_evaluation.py    # Advanced evaluation and visualizations
│   └── main.py                   # Pipeline orchestrator
├── data/
│   └── bank_data.csv            # Dataset file
├── models/                      # Saved model files (.pkl)
├── results/                     # Output files, plots, reports
├── requirements.txt             # Project dependencies
├── run_enhanced_pipeline.py     # Enhanced pipeline runner script
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
   
   This will install:
   - Core libraries: pandas, numpy, scikit-learn
   - Advanced ML: xgboost, lightgbm
   - Optimization: optuna
   - Visualization: matplotlib, seaborn

4. **Prepare data:**
   - Place `bank_data.csv` in the `data/` directory
   - Ensure the file uses semicolon (;) as delimiter

## Quick Start

### Option 1: Full Enhanced Pipeline ⭐ (Recommended)
**Best results with comprehensive analysis**

```bash
cd /path/to/bank-term-deposit-prediction
source bank_env/bin/activate
python src/main.py
```

- ✅ Trains 8 models (XGBoost, LightGBM, Random Forest, SVM, etc.)
- ✅ Hyperparameter tuning with Optuna (20 trials per model)
- ✅ Threshold optimization for each model
- ✅ Model stacking (ensemble)
- ✅ Advanced evaluation with visualizations
- ⏱️ **Time: 10-15 minutes**

### Option 2: Fast Test Pipeline ⚡
**Quick verification without slow models**

```bash
python run_fast_pipeline.py
```

- ✅ Trains 7 models (excludes SVM)
- ✅ Reduced tuning (10 trials)
- ⏱️ **Time: 5-8 minutes**

### Option 3: Basic Pipeline 📚
**Simple models for learning**

```python
from src.main import main_basic
results = main_basic()
```

- ✅ Naive Bayes and Decision Trees only
- ✅ No hyperparameter tuning
- ⏱️ **Time: 1-2 minutes**

## Usage Guide

### Running the Pipeline

**Command Line (Recommended)**
```bash
# Activate environment
source bank_env/bin/activate  # On Windows: bank_env\Scripts\activate

# Run full pipeline
python src/main.py
```

**Alternative Methods**
```bash
# Using runner script
python run_enhanced_pipeline.py

# Using Python import
python -c "from src.main import main_enhanced; main_enhanced()"
```

**In Jupyter Notebook or Python Script**
```python
from src.main import main_enhanced

# Run complete pipeline
results = main_enhanced()

# Access best model
best_model = results['best_model']
print(f"Best model: {results['best_model_name']}")
```

### What You'll Get

After running the pipeline, check these directories:

**📁 `models/` - Trained Models**
```
models/
├── naive_bayes_enhanced_20251024_211125.pkl
├── random_forest_enhanced_20251024_211125.pkl
├── random_forest_tuned_20251024_211125.pkl
├── xgboost_enhanced_20251024_211125.pkl
├── xgboost_tuned_20251024_211125.pkl
├── lightgbm_enhanced_20251024_211125.pkl
├── lightgbm_tuned_20251024_211125.pkl
├── stacking_voting_20251024_211125.pkl
└── best_thresholds_20251024_211125.pkl
```

**📁 `results/` - Visualizations**
```
results/
├── enhanced_model_comparison.png      # Performance bar charts
├── roc_curves_comparison.png          # ROC curves for all models
├── precision_recall_curves.png        # Precision-Recall analysis
├── enhanced_confusion_matrices.png    # Confusion matrices grid
├── threshold_optimization.png         # Threshold analysis
├── calibration_curves.png             # Model calibration
├── correlation_matrix.png             # Feature correlations
└── subscription_rates_by_category.png # EDA insights
```

### Expected Output

When you run the pipeline, you'll see progress like this:

```
 BANK TERM DEPOSIT PREDICTION - ENHANCED ML PIPELINE
====================================================================

 STEP 1: DATA PROCESSING
   ✓ Data loaded successfully: (41188, 21)
   ✓ Target distribution: No=88.3%, Yes=11.7%

 STEP 2: FEATURE ENGINEERING
   ✓ Created 'pdays999' feature
   ✓ Removed correlated features
   ✓ Final feature matrix: (41188, 51)

 STEP 3: ENHANCED MODEL TRAINING
   
   Processing XGBOOST (4/8)
   ✓ Training xgboost... (0.6s)
     Accuracy: 0.9175, F1-Score: 0.6033, ROC-AUC: 0.9479
   
   [TUNING] Hyperparameter optimization for xgboost...
   Running 20 optimization trials (this may take a few minutes)...
   Trial 20/20 complete | Best roc_auc: 0.9520
   ✓ Optimization completed in 11.5 seconds
   
   [THRESHOLD] Optimizing decision threshold...
   ✓ Best threshold: 0.390 (F1-Score: 0.6033)

 STEP 4: ENHANCED MODEL EVALUATION
   ✓ ROC curves saved
   ✓ Precision-Recall curves saved
   ✓ Confusion matrices saved
   ✓ Threshold optimization plots saved

 ENHANCED PIPELINE COMPLETED SUCCESSFULLY

Best Overall Model: xgboost
  Accuracy: 91.75% | F1-Score: 60.33% | ROC-AUC: 94.79%
```

### Troubleshooting

**Issue: Import errors or module not found**
```bash
# Solution: Ensure environment is activated and dependencies installed
source bank_env/bin/activate
pip install -r requirements.txt
```

**Issue: Data file not found**
```bash
# Solution: Verify data file location and format
ls data/bank_data.csv  # Should exist
head -n 1 data/bank_data.csv  # Should show semicolon-separated columns
```

**Issue: Pipeline takes too long**
```bash
# Solution 1: Use fast pipeline (excludes SVM)
python run_fast_pipeline.py

# Solution 2: Reduce tuning trials in src/config.py
# Change: 'n_trials': 10  # Instead of 20
```

**Issue: Memory errors**
```python
# Solution: Disable some models in src/config.py
# Comment out SVM (slowest/most memory-intensive):
MODEL_CONFIG = {
    # 'svm': { ... },  # Commented out
}
```

### Customization

Edit `src/config.py` to customize the pipeline:

**Speed up execution:**
```python
TUNING_CONFIG = {
    'enable_tuning': False,  # Skip hyperparameter tuning
    'n_trials': 10,          # Or reduce trials (instead of 20)
    'cv_folds': 2,           # Reduce cross-validation folds
}
```

**Optimize for specific metric:**
```python
THRESHOLD_CONFIG = {
    'optimization_metric': 'precision',  # Options: 'f1', 'precision', 'recall'
}
```

**Disable features:**
```python
STACKING_CONFIG = {
    'enable_stacking': False,  # Skip ensemble creation
}
```

### Advanced Usage

**Run Individual Pipeline Components**
```python
from src.data_processing import main_data_processing
from src.feature_engineering import main_feature_engineering
from src.enhanced_model_training import main_enhanced_model_training
from src.enhanced_evaluation import main_enhanced_evaluation

# Step 1: Load and explore data
df, summary = main_data_processing()

# Step 2: Engineer features
processed_data = main_feature_engineering(df)

# Step 3: Train models with tuning and stacking
training_results = main_enhanced_model_training(processed_data)

# Step 4: Evaluate models with advanced metrics
evaluation_results = main_enhanced_evaluation(
    training_results['training_results'],
    processed_data['X_test'],
    processed_data['y_test']
)
```

**Load and Use Trained Models**
```python
import joblib

# Load saved model
model = joblib.load('models/xgboost_enhanced_20251024_211125.pkl')

# Make predictions
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]
```

## FAQ

**Q: Which pipeline should I run first?**  
A: Start with `run_fast_pipeline.py` to verify everything works (~5-8 min), then run `python src/main.py` for best results (~10-15 min).

**Q: How long does the full pipeline take?**  
A: Approximately 10-15 minutes on a modern laptop. The fast pipeline takes 5-8 minutes.

**Q: Can I stop the pipeline midway?**  
A: Yes, press `Ctrl+C`. Already trained models are saved. You can restart anytime.

**Q: Which model should I use in production?**  
A: **XGBoost** achieved the best F1-score (60.33%) with balanced precision/recall. **LightGBM Tuned** has the best ROC-AUC (95.22%) for probability ranking.

**Q: What if I get memory errors?**  
A: Comment out SVM in `src/config.py` (most memory-intensive) or reduce the dataset size for testing.

**Q: How do I update the models with new data?**  
A: Replace `data/bank_data.csv` with your new data (same format) and re-run the pipeline. Models will be saved with new timestamps.

**Q: Can I use this with my own dataset?**  
A: Yes! The pipeline is modular. You'll need to adjust feature engineering in `src/feature_engineering.py` to match your data schema.

## Key Insights & Findings

### Data Insights
- **Class Imbalance**: ~88% customers do not subscribe to term deposits
- **High Correlation**: Strong correlation between economic indicators (emp.var.rate, euribor3m, nr.employed)
- **Seasonal Patterns**: March and December show higher subscription rates
- **Contact Method**: Cellular contact generally outperforms telephone

### Model Performance
The enhanced pipeline implements and compares 8+ algorithms:

#### Basic Models
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

#### Advanced Models
4. **Random Forest**
   - Ensemble of decision trees
   - Reduces overfitting
   - Excellent feature importance

5. **XGBoost**
   - Gradient boosting framework
   - High performance on structured data
   - Built-in regularization

6. **LightGBM**
   - Fast gradient boosting
   - Memory efficient
   - Good for large datasets

7. **Support Vector Machine (SVM)**
   - Effective for high-dimensional data
   - Robust to outliers
   - Probability estimates available

8. **Logistic Regression**
   - Linear baseline model
   - Interpretable coefficients
   - Fast training and prediction

#### Ensemble Methods
9. **Model Stacking**
   - Combines multiple models
   - Voting classifier with soft voting
   - Often achieves best performance

### Business Recommendations
1. **Target Segments**: Focus campaigns on students and retired individuals
2. **Timing**: Schedule major campaigns in March and December
3. **Contact Method**: Prioritize cellular over telephone contact
4. **Previous Success**: Heavily weight previous campaign success in targeting decisions

## Enhanced Pipeline Features

### What Was Added

The project now includes an advanced pipeline that extends the basic models with professional ML techniques:

**1. Additional Models**
- Random Forest: Ensemble method with 100+ decision trees
- XGBoost: Gradient boosting for high performance
- LightGBM: Fast gradient boosting, memory efficient
- SVM: Support Vector Machine for complex decision boundaries
- Logistic Regression: Linear baseline for comparison

**2. Hyperparameter Tuning**
- Automated optimization using Optuna (Bayesian optimization)
- 10-20 trials per model with cross-validation
- Searches parameter spaces intelligently
- Typical improvement: 2-5% in ROC-AUC

**3. Threshold Optimization**
- Tests 80+ different probability thresholds
- Finds optimal cutoff for classification
- Can optimize for F1-score, precision, or recall
- Visualizes metric changes across thresholds

**4. Model Stacking**
- Voting ensemble combining top models
- Uses soft voting (probability averaging)
- Often achieves better performance than individual models
- Automatically selects models with probability estimates

**5. Advanced Evaluation**
- ROC curves with AUC scores for all models
- Precision-Recall curves (better for imbalanced data)
- Calibration plots (how well probabilities match reality)
- Threshold analysis visualization
- Comprehensive confusion matrices

### Results Achieved

Based on actual pipeline run:
- **Best F1-Score**: 60.33% (XGBoost)
- **Best ROC-AUC**: 95.22% (LightGBM Tuned)
- **Best Precision**: 70.92% (LightGBM Tuned)
- **Best Accuracy**: 91.90% (XGBoost Tuned)

The stacking ensemble achieved F1=58.15% and ROC-AUC=94.72%, demonstrating effective model combination.

## Technical Highlights

### Feature Engineering
- **pdays999 Feature**: Binary indicator for customers with no previous contact
- **Multicollinearity Handling**: Removes highly correlated economic indicators
- **Categorical Encoding**: One-hot encoding with drop_first=True
- **Data Validation**: Comprehensive missing value and data quality checks

### Model Training
- **Label Encoding**: Automatic handling for XGBoost/LightGBM requirements
- **Progress Tracking**: Real-time progress with trial counters and time estimates
- **Error Handling**: Graceful handling of model-specific requirements
- **Parallel Processing**: Uses all CPU cores where applicable

### Model Evaluation
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Average Precision, Log Loss
- **Advanced Visualizations**: ROC curves, Precision-Recall curves, Calibration plots, Threshold optimization
- **Binary Label Conversion**: Proper handling for probability-based metrics
- **Statistical Reports**: Detailed classification reports for each model
- **Model Persistence**: Automated saving of trained and tuned models with timestamps

### Code Quality
- **Modular Design**: Separates concerns across logical modules
- **Error Handling**: Robust exception handling throughout pipeline
- **Progress Tracking**: Real-time updates with time estimates
- **Documentation**: Clear docstrings explaining each function

## How the Enhanced Features Work

### Hyperparameter Tuning

The pipeline uses Optuna for intelligent hyperparameter search:

1. Defines search space for each model (e.g., tree depth, learning rate)
2. Runs multiple trials testing different parameter combinations
3. Uses cross-validation to evaluate each combination
4. Selects parameters that maximize ROC-AUC score
5. Trains final model with best parameters

Example output:
```
[TUNING] Hyperparameter optimization for xgboost...
Running 20 optimization trials (this may take a few minutes)...
Trial 1/20 complete | Best roc_auc: 0.9401
...
Trial 20/20 complete | Best roc_auc: 0.9435
Optimization completed in 24.5 seconds
Best parameters: {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.01}
```

### Threshold Optimization

Instead of using the default 0.5 threshold for classification:

1. Tests thresholds from 0.1 to 0.9 in 0.01 increments
2. Calculates F1-score (or other metric) at each threshold
3. Selects threshold that maximizes the chosen metric
4. Saves optimal threshold for each model

This can significantly improve results, especially for imbalanced datasets.

### Model Stacking

The voting ensemble combines predictions from multiple models:

1. Collects all models that can produce probability estimates
2. For each prediction, gets probabilities from all base models
3. Averages the probabilities (soft voting)
4. Makes final prediction using the averaged probability

This typically produces more robust predictions than any single model.

### Performance Comparison

| Feature | Basic Pipeline | Enhanced Pipeline |
|---------|---------------|-------------------|
| Models | 3 | 8+ |
| Hyperparameter Tuning | No | Yes (Optuna) |
| Threshold Optimization | No | Yes |
| Model Stacking | No | Yes |
| Execution Time | 1-2 min | 10-15 min |
| Best F1-Score | ~52% | ~60% |
| Best ROC-AUC | ~73% | ~95% |

## Results & Outputs

The pipeline generates several outputs:

### Models
- **Saved Models**: Serialized model files in `models/` directory
- **Training Results**: Comprehensive training metrics and metadata

### Visualizations
- **Target Distribution**: Class balance visualization
- **Subscription Rates**: Analysis by categorical variables
- **Correlation Matrix**: Feature correlation heatmaps
- **Enhanced Model Comparison**: Multi-metric performance comparison
- **ROC Curves**: Model discrimination analysis with AUC scores
- **Precision-Recall Curves**: Performance analysis for imbalanced data
- **Confusion Matrices**: Model performance visualization
- **Threshold Optimization**: Dynamic threshold analysis plots
- **Calibration Curves**: Model probability calibration analysis

### Reports
- **Model Comparison**: Side-by-side performance metrics
- **Feature Importance**: Key predictive features (for tree models)
- **Evaluation Summary**: Comprehensive model assessment

## Future Enhancements

### Model Improvements
- **Deep Learning**: Neural networks with TensorFlow/PyTorch
- **Advanced Ensemble**: Blending, bagging, and boosting combinations
- **Class Balancing**: SMOTE, class weights, or cost-sensitive learning
- **AutoML**: Automated model selection and feature engineering

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
- **xgboost**: Gradient boosting framework
- **lightgbm**: Fast gradient boosting
- **optuna**: Hyperparameter optimization
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
