"""
Data processing module for Bank Term Deposit Prediction.
Handles data loading, cleaning, and exploratory data analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Import from config - absolute import
from config import (
    DATA_CONFIG, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS,
    VIZ_CONFIG, RESULTS_DIR, DATA_DIR, ANALYSIS_CONFIG
)

# Set pandas display options
pd.set_option('display.max_columns', ANALYSIS_CONFIG['max_columns_display'])

class BankDataProcessor:
    """Data processor for bank marketing campaign dataset."""

    def __init__(self, data_path: Path = None):
        """Initialize the data processor."""
        self.data_path = data_path or DATA_DIR / DATA_CONFIG['file_name']
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """Load the bank marketing dataset."""
        try:
            self.df = pd.read_csv(
                self.data_path,
                sep=DATA_CONFIG['separator']
            )
            print(f"Data loaded successfully: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            return self.df
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

    def basic_info(self) -> Dict[str, Any]:
        """Display basic information about the dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'null_values': self.df.isnull().sum().to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum()
        }

        print("=== DATASET BASIC INFORMATION ===")
        print(f"Shape: {info['shape']}")
        print(f"Columns: {len(info['columns'])}")
        print(f"Memory Usage: {info['memory_usage']:,} bytes")
        print("\nFirst 5 rows:")
        print(self.df.head())
        print("\nData Types:")
        print(self.df.dtypes)
        print("\nNull Values:")
        null_counts = self.df.isnull().sum()
        if null_counts.sum() == 0:
            print("No null values found")
        else:
            print(null_counts[null_counts > 0])

        return info

    def analyze_target_variable(self) -> Dict[str, Any]:
        """Analyze the target variable distribution."""
        target_col = DATA_CONFIG['target_column']

        # Value counts
        value_counts = self.df[target_col].value_counts()
        value_percentages = self.df[target_col].value_counts(normalize=True) * 100

        print("=== TARGET VARIABLE ANALYSIS ===")
        print(f"Target variable '{target_col}' distribution:")
        print(value_counts)
        print("\nPercentages:")
        for value, pct in value_percentages.items():
            print(f"{value}: {pct:.2f}%")

        # Create visualization
        plt.figure(figsize=VIZ_CONFIG['figure_size'])
        value_counts.plot(
            kind="bar",
            color=VIZ_CONFIG['subscription_colors']
        )
        plt.title("Distribution of Target Variable (Term Deposit Subscription)")
        plt.xlabel("Subscription to Term Deposit")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "target_distribution.png", dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.close()

        return {
            'value_counts': value_counts.to_dict(),
            'percentages': value_percentages.to_dict()
        }

    def subscription_rate_analysis(self) -> Dict[str, pd.DataFrame]:
        """Analyze subscription rates by categorical variables."""
        # Get categorical columns that exist in the dataset
        cat_cols = []
        for col in CATEGORICAL_COLUMNS:
            if col in self.df.columns:
                cat_cols.append(col)
            else:
                print(f"Warning: Column '{col}' not found in dataset")

        target_col = DATA_CONFIG['target_column']
        subscription_rates = {}

        print("=== SUBSCRIPTION RATE ANALYSIS ===")

        for col in cat_cols:
            print(f"\nAnalyzing {col}:")
            print(f"Unique values: {self.df[col].nunique()}")

            # Calculate subscription rate
            try:
                rate = pd.crosstab(self.df[col], self.df[target_col], normalize='index') * 100
                subscription_rates[col] = rate
                print(f"Subscription rate by {col}:")
                print(rate.round(2))
            except Exception as e:
                print(f"Error analyzing {col}: {e}")
                continue

        # Create visualizations
        self._plot_subscription_rates(subscription_rates)

        return subscription_rates

    def _plot_subscription_rates(self, subscription_rates: Dict[str, pd.DataFrame]):
        """Create visualizations for subscription rates."""
        cat_cols = list(subscription_rates.keys())
        n_cols = len(cat_cols)

        if n_cols == 0:
            print("No categorical columns to plot")
            return

        # Calculate subplot grid
        n_subplot_cols = min(2, n_cols)
        n_rows = (n_cols + n_subplot_cols - 1) // n_subplot_cols

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_subplot_cols,
            figsize=(9 * n_subplot_cols, 5 * n_rows)
        )

        # Handle single subplot case
        if n_rows == 1 and n_subplot_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = list(axes)
        else:
            axes = axes.flatten()

        for i, col in enumerate(cat_cols):
            if 'yes' in subscription_rates[col].columns:
                subscription_rates[col]['yes'].plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'Subscription Rate by {col}')
                axes[i].set_ylim(0, max(70, subscription_rates[col]['yes'].max() * 1.1))
                axes[i].set_ylabel('Subscription Rate (%)')
                axes[i].tick_params(axis='x', rotation=45)

        # Hide empty subplots
        for i in range(len(cat_cols), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout(pad=2.0)
        plt.savefig(RESULTS_DIR / "subscription_rates_by_category.png", dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        print(f"Subscription rate visualizations saved to {RESULTS_DIR}/subscription_rates_by_category.png")

    def correlation_analysis(self) -> Dict[str, Any]:
        """Perform correlation analysis on numerical features."""
        # Get numerical columns that exist in the dataset
        num_cols = []
        for col in NUMERICAL_COLUMNS:
            if col in self.df.columns:
                num_cols.append(col)
            else:
                print(f"Warning: Numerical column '{col}' not found in dataset")

        if not num_cols:
            print("No numerical columns found for correlation analysis")
            return {}

        num_df = self.df[num_cols]

        # Calculate correlation matrix
        corr_matrix = num_df.corr(method=ANALYSIS_CONFIG['correlation_method'])

        print("=== CORRELATION ANALYSIS ===")
        print("Numerical columns analyzed:")
        print(num_cols)

        # Create correlation heatmap
        plt.figure(figsize=(12, 12))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.3f'
        )
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "correlation_matrix.png", dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.close()

        # Identify highly correlated features
        high_corr_pairs = self._find_high_correlations(corr_matrix)

        return {
            'correlation_matrix': corr_matrix,
            'high_correlations': high_corr_pairs,
            'numerical_columns': num_cols
        }

    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.8) -> list:
        """Find pairs of highly correlated features."""
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = abs(corr_matrix.iloc[i, j])
                if corr_value >= threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })

        if high_corr_pairs:
            print("\nHighly correlated feature pairs (|r| >= 0.8):")
            for pair in high_corr_pairs:
                print(f"{pair['feature1']} - {pair['feature2']}: {pair['correlation']:.3f}")
        else:
            print("\nNo highly correlated feature pairs found (|r| >= 0.8)")

        return high_corr_pairs

    def generate_data_summary(self) -> Dict[str, Any]:
        """Generate comprehensive data summary."""
        summary = {
            'basic_info': self.basic_info(),
            'target_analysis': self.analyze_target_variable(),
            'subscription_rates': self.subscription_rate_analysis(),
            'correlation_analysis': self.correlation_analysis()
        }

        # Generate insights
        insights = self._generate_insights(summary)
        summary['insights'] = insights

        return summary

    def _generate_insights(self, summary: Dict[str, Any]) -> list:
        """Generate key insights from the analysis."""
        insights = []

        # Target distribution insights
        target_analysis = summary['target_analysis']
        if 'no' in target_analysis['percentages'] and 'yes' in target_analysis['percentages']:
            no_pct = target_analysis['percentages']['no']
            yes_pct = target_analysis['percentages']['yes']
            insights.append(f"Dataset is imbalanced: {no_pct:.1f}% no subscription vs {yes_pct:.1f}% yes")

        # Correlation insights
        high_corr = summary['correlation_analysis'].get('high_correlations', [])
        if high_corr:
            insights.append(f"Found {len(high_corr)} highly correlated feature pairs - consider feature selection")

        # General insights based on typical banking data patterns
        insights.extend([
            "Consider targeting students and retired people for higher conversion rates",
            "March and December typically show higher campaign success rates",
            "Cellular contact method generally outperforms telephone",
            "Previous campaign success is strong predictor of future success"
        ])

        return insights

def main_data_processing(data_path: Path = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Main function to execute data processing pipeline."""
    print("="*60)
    print("BANK TERM DEPOSIT PREDICTION - DATA PROCESSING")
    print("="*60)

    # Initialize processor
    processor = BankDataProcessor(data_path)

    # Load data
    df = processor.load_data()

    # Generate comprehensive analysis
    summary = processor.generate_data_summary()

    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    for insight in summary['insights']:
        print(f"• {insight}")

    print(f"\nData processing completed. Results saved to: {RESULTS_DIR}")
    print("="*60)

    return df, summary

if __name__ == "__main__":
    df, summary = main_data_processing()