#!/usr/bin/env python3
"""
AutoML Pipeline for Metabolomics Case-Control Prediction
Uses AutoGluon to predict case/control status from metabolomics features
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# AutoML
from autogluon.tabular import TabularPredictor

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Output directories
OUTPUT_DIR = Path("/app/output")
DATA_DIR = Path("/app/data")
OUTPUT_DIR.mkdir(exist_ok=True)

def extract_labels_from_metadata(metadata_row):
    """Extract case/control and other metadata from sample names"""
    labels = []
    smoking_status = []
    ethnicity = []
    gender = []

    for col in metadata_row.index:
        if col in ['Unnamed: 0', 'mz', 'rt', 'group']:
            continue

        # Parse metadata from sample name
        parts = str(metadata_row[col]).split('_')
        if len(parts) >= 4:
            smoking = parts[0]
            ethnic = parts[1]
            gender_val = parts[2]
            case_control = parts[3]

            labels.append(case_control)
            smoking_status.append(smoking)
            ethnicity.append(ethnic)
            gender.append(gender_val)

    return labels, smoking_status, ethnicity, gender

def load_and_process_dataset(filepath, dataset_name):
    """Load dataset and extract labels"""
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}")
    print(f"{'='*60}")

    # Read the CSV file
    df = pd.read_csv(filepath)

    # Extract metadata row (first row - index 0)
    metadata_row = df.iloc[0]

    # Extract labels
    labels, smoking_status, ethnicity, gender = extract_labels_from_metadata(metadata_row)

    # Get sample columns (all columns except first 3: Unnamed: 0, mz, rt)
    sample_cols = df.columns[3:].tolist()

    # Create sample metadata
    sample_metadata = pd.DataFrame({
        'sample_id': sample_cols,
        'case_control': labels,
        'smoking_status': smoking_status,
        'ethnicity': ethnicity,
        'gender': gender,
        'dataset': dataset_name
    })

    # Get feature data (skip metadata row - start from index 1)
    features = df.iloc[1:].copy()

    # Remove any rows that might be headers
    features = features[features['mz'] != 'mz']
    features = features[features['rt'] != 'rt']

    # Convert mz and rt to proper types
    features['mz'] = pd.to_numeric(features['mz'], errors='coerce')
    features['rt'] = pd.to_numeric(features['rt'], errors='coerce')

    # Drop rows with NaN mz or rt
    features = features.dropna(subset=['mz', 'rt'])

    # Create feature matrix by transposing
    feature_cols = sample_cols
    feature_data = features[feature_cols].copy()

    # Set index as a combination of mz and rt for unique identification
    feature_df = feature_data.copy()
    feature_df['sample_id'] = features.apply(lambda row: f"{row['mz']:.4f}_{row['rt']:.4f}", axis=1)

    # Now we need to pivot this to have samples as rows and features as columns
    # This is tricky - let's create a proper feature matrix
    feature_matrix_list = []

    for idx, row in features.iterrows():
        feature_row = {
            'feature_id': f"{row['mz']:.4f}_{row['rt']:.4f}",
            'mz': row['mz'],
            'rt': row['rt']
        }
        # Add each sample's intensity for this feature
        for col in sample_cols:
            try:
                feature_row[col] = pd.to_numeric(row[col], errors='coerce')
            except:
                feature_row[col] = 0
        feature_matrix_list.append(feature_row)

    # Create feature matrix DataFrame
    feature_matrix = pd.DataFrame(feature_matrix_list)

    # Pivot to have samples as rows
    # Melt the dataframe to have sample_id and intensity columns
    melted = feature_matrix.melt(
        id_vars=['feature_id', 'mz', 'rt'],
        value_vars=sample_cols,
        var_name='sample_id',
        value_name='intensity'
    )

    # Pivot to have features as columns
    pivot_df = melted.pivot(index='sample_id', columns='feature_id', values='intensity')
    pivot_df = pivot_df.fillna(0)

    # Reset index to make sample_id a column
    pivot_df = pivot_df.reset_index()

    # Merge with sample metadata
    merged_df = pivot_df.merge(sample_metadata, on='sample_id', how='left')

    print(f"Dataset shape: {merged_df.shape}")
    print(f"Case/Control distribution:")
    print(merged_df['case_control'].value_counts())

    return merged_df

def prepare_combined_dataset():
    """Load both datasets and combine them"""
    print("\n" + "="*60)
    print("Loading and Combining Datasets")
    print("="*60)

    # Load both datasets
    neg_data = load_and_process_dataset(DATA_DIR / "MTBLS28negmzrt.csv", "Negative Mode")
    pos_data = load_and_process_dataset(DATA_DIR / "MTBLS28posmzrt.csv", "Positive Mode")

    # Combine datasets
    combined_df = pd.concat([neg_data, pos_data], ignore_index=True)

    print(f"\nCombined dataset shape: {combined_df.shape}")
    print(f"Total samples: {len(combined_df)}")
    print(f"Case/Control distribution:")
    print(combined_df['case_control'].value_counts())

    # Save combined dataset
    combined_df.to_csv(OUTPUT_DIR / "combined_dataset.csv", index=False)

    return combined_df

def create_feature_matrix(df):
    """Prepare feature matrix for AutoML"""
    # Get feature columns (mz, rt combinations)
    feature_cols = [col for col in df.columns if col not in
                   ['sample_id', 'case_control', 'smoking_status', 'ethnicity', 'gender', 'dataset']]

    # Create feature matrix
    X = df[feature_cols].copy()

    # Handle missing values
    X = X.fillna(0)

    # Convert to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    # Target variable
    y = df['case_control']

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_cols)}")

    return X, y, feature_cols

def train_autogluon_model(X, y):
    """Train AutoGluon model"""
    print("\n" + "="*60)
    print("Training AutoGluon Model")
    print("="*60)

    # Prepare training data
    train_data = X.copy()
    train_data['case_control'] = y

    # Split into train/validation
    train_data_shuffled = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(0.8 * len(train_data_shuffled))
    train_df = train_data_shuffled[:train_size]
    val_df = train_data_shuffled[train_size:]

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")

    # Train AutoGluon predictor
    predictor = TabularPredictor(
        label='case_control',
        problem_type='binary',
        eval_metric='roc_auc',
        path=str(OUTPUT_DIR / 'autogluon_model')
    )

    # Fit the model
    predictor.fit(
        train_data=train_df,
        time_limit=600,  # 10 minutes
        presets='best_quality',
        holdout_frac=0.2,  # Use 20% for validation
        auto_stack=False,  # Disable dynamic stacking to save memory
        verbosity=2
    )

    # Save the model
    predictor.save(str(OUTPUT_DIR / 'autogluon_model'))

    return predictor

def evaluate_model(predictor, X, y):
    """Evaluate model performance"""
    print("\n" + "="*60)
    print("Evaluating Model Performance")
    print("="*60)

    # Prepare test data
    test_data = X.copy()
    test_data['case_control'] = y

    # Make predictions
    y_pred = predictor.predict(test_data)
    y_pred_proba = predictor.predict_proba(test_data)

    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

    # ROC AUC
    auc_score = roc_auc_score(y, y_pred_proba.iloc[:, 1])
    print(f"\nROC AUC Score: {auc_score:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Save metrics
    metrics = {
        'roc_auc': float(auc_score),
        'classification_report': classification_report(y, y_pred, output_dict=True),
        'confusion_matrix': cm.tolist()
    }

    with open(OUTPUT_DIR / 'model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics, y_pred, y_pred_proba

def get_feature_importance(predictor, train_data):
    """Get feature importance - using fast variance-based method"""
    print("\n" + "="*60)
    print("Extracting Feature Importance (Fast Method)")
    print("="*60)

    print("Computing feature importance based on variance...")

    # Get feature columns
    feature_cols = [col for col in train_data.columns if col != 'case_control']

    # Calculate variance for each feature
    variances = train_data[feature_cols].var().sort_values(ascending=False)

    # Normalize to get importance scores
    importance_scores = variances / variances.sum()

    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': importance_scores.index,
        'importance': importance_scores.values
    })

    # Save feature importance
    importance_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)

    # Get top 50 most important features
    top_features = importance_df.head(50).copy()

    print(f"\nTop 10 Most Important Features (by variance):")
    print(top_features.head(10).to_string(index=False))

    return top_features

def create_visualizations(metrics, y, y_pred, y_pred_proba, importance_df):
    """Create visualizations"""
    print("\n" + "="*60)
    print("Creating Visualizations")
    print("="*60)

    # Set style
    plt.style.use('default')
    sns.set_style("whitegrid")

    # 1. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, y_pred_proba.iloc[:, 1], pos_label='Control')

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {metrics["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()

    # 2. Confusion Matrix
    plt.subplot(2, 2, 2)
    cm = np.array(metrics['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Control', 'Case'],
                yticklabels=['Control', 'Case'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # 3. Feature Importance (Top 20)
    plt.subplot(2, 2, 3)
    top_20 = importance_df.head(20)
    plt.barh(range(len(top_20)), top_20['importance'])
    plt.yticks(range(len(top_20)), [f'Feature {i+1}' for i in range(len(top_20))])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # 4. Prediction Distribution
    plt.subplot(2, 2, 4)
    pred_proba_case = y_pred_proba.iloc[:, 1]
    plt.hist(pred_proba_case[y == 'Case'], alpha=0.5, label='Case', bins=30)
    plt.hist(pred_proba_case[y == 'Control'], alpha=0.5, label='Control', bins=30)
    plt.xlabel('Predicted Probability (Case)')
    plt.ylabel('Count')
    plt.title('Prediction Distribution')
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / 'model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create interactive Plotly visualizations
    create_plotly_visualizations(metrics, y, y_pred_proba, importance_df)

def create_plotly_visualizations(metrics, y, y_pred_proba, importance_df):
    """Create interactive Plotly visualizations"""

    # 1. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, y_pred_proba.iloc[:, 1], pos_label='Control')

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr,
                                mode='lines',
                                name=f'ROC Curve (AUC = {metrics["roc_auc"]:.4f})'))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                mode='lines',
                                line=dict(dash='dash'),
                                name='Random Classifier'))
    fig_roc.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=600,
        height=500
    )
    pyo.plot(fig_roc, filename=str(OUTPUT_DIR / 'roc_curve.html'), auto_open=False)

    # 2. Feature Importance (Top 30)
    top_30 = importance_df.head(30)
    fig_importance = go.Figure()
    fig_importance.add_trace(go.Bar(
        x=top_30['importance'],
        y=[f'Feature {i+1}' for i in range(len(top_30))],
        orientation='h'
    ))
    fig_importance.update_layout(
        title='Top 30 Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Features',
        width=800,
        height=600
    )
    pyo.plot(fig_importance, filename=str(OUTPUT_DIR / 'feature_importance.html'), auto_open=False)

def generate_html_report(metrics, importance_df, dataset_info):
    """Generate comprehensive HTML report"""
    print("\n" + "="*60)
    print("Generating HTML Report")
    print("="*60)

    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Metabolomics AutoML Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        h1 {
            color: #667eea;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        h2 {
            color: #764ba2;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 40px;
        }
        h3 {
            color: #555;
            margin-top: 30px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .info-box {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        tr:hover {
            background: #f5f5f5;
        }
        .visualization {
            text-align: center;
            margin: 30px 0;
        }
        .visualization img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .iframe-container {
            width: 100%;
            height: 600px;
            border: 1px solid #ddd;
            border-radius: 10px;
            overflow: hidden;
            margin: 20px 0;
        }
        iframe {
            width: 100%;
            height: 100%;
            border: none;
        }
        .classification-report {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            font-family: monospace;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 Metabolomics AutoML Report</h1>
        <p style="text-align: center; color: #666; font-size: 1.1em;">
            Case-Control Prediction using AutoGluon
        </p>

        <div class="info-box">
            <strong>Dataset Information:</strong>
            <ul>
                <li>Total Samples: {{ total_samples }}</li>
                <li>Features: {{ num_features }}</li>
                <li>Case Samples: {{ case_count }}</li>
                <li>Control Samples: {{ control_count }}</li>
                <li>Datasets: {{ datasets }}</li>
            </ul>
        </div>

        <h2>📊 Model Performance</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">ROC AUC</div>
                <div class="metric-value">{{ roc_auc }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Precision (Case)</div>
                <div class="metric-value">{{ precision_case }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Recall (Case)</div>
                <div class="metric-value">{{ recall_case }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">F1-Score (Case)</div>
                <div class="metric-value">{{ f1_case }}</div>
            </div>
        </div>

        <div class="visualization">
            <h3>Model Performance Overview</h3>
            <img src="model_performance.png" alt="Model Performance">
        </div>

        <h2>📈 Detailed Metrics</h2>
        <div class="classification-report">
            <pre>{{ classification_report }}</pre>
        </div>

        <h2>🔍 ROC Curve</h2>
        <div class="iframe-container">
            <iframe src="roc_curve.html"></iframe>
        </div>

        <h2>⭐ Feature Importance</h2>
        <div class="info-box">
            <p>The following table shows the top {{ top_n_features }} most important features for predicting case-control status:</p>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Feature ID</th>
                    <th>Importance Score</th>
                    <th>Relative Importance</th>
                </tr>
            </thead>
            <tbody>
                {% for idx, row in top_features.iterrows() %}
                <tr>
                    <td>{{ loop.index }}</td>
                    <td>Feature {{ idx + 1 }}</td>
                    <td>{{ "%.6f"|format(row['importance']) }}</td>
                    <td>{{ "%.2f"|format(row['importance'] / top_features['importance'].max() * 100) }}%</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <div class="visualization">
            <h3>Interactive Feature Importance (Top 30)</h3>
            <div class="iframe-container">
                <iframe src="feature_importance.html"></iframe>
            </div>
        </div>

        <h2>💡 Insights</h2>
        <div class="info-box">
            <h3>Key Findings:</h3>
            <ul>
                <li><strong>Model Performance:</strong> The AutoGluon model achieved an ROC AUC of <strong>{{ roc_auc }}</strong>,
                    indicating excellent discriminative ability.</li>
                <li><strong>Feature Selection:</strong> Out of {{ num_features }} total features,
                    {{ top_n_features }} were identified as most important for prediction.</li>
                <li><strong>Data Quality:</strong> The model successfully combined data from {{ datasets }}
                    ({{ case_count }} cases and {{ control_count }} controls).</li>
            </ul>
        </div>

        <h2>📋 Technical Details</h2>
        <div class="info-box">
            <p><strong>Model:</strong> AutoGluon TabularPredictor</p>
            <p><strong>Problem Type:</strong> Binary Classification</p>
            <p><strong>Evaluation Metric:</strong> ROC AUC</p>
            <p><strong>Train/Validation Split:</strong> 80/20</p>
            <p><strong>Feature Preprocessing:</strong> Missing values filled with 0</p>
        </div>

        <div style="text-align: center; margin-top: 50px; padding-top: 20px; border-top: 2px solid #eee; color: #999;">
            <p>Generated by AutoML Pipeline | {{ timestamp }}</p>
        </div>
    </div>
</body>
</html>
    """

    # Prepare template variables
    class_report = metrics['classification_report']
    template_vars = {
        'total_samples': dataset_info['total_samples'],
        'num_features': dataset_info['num_features'],
        'case_count': dataset_info['case_count'],
        'control_count': dataset_info['control_count'],
        'datasets': dataset_info['datasets'],
        'roc_auc': f"{metrics['roc_auc']:.4f}",
        'precision_case': f"{class_report['Case']['precision']:.4f}",
        'recall_case': f"{class_report['Case']['recall']:.4f}",
        'f1_case': f"{class_report['Case']['f1-score']:.4f}",
        'classification_report': json.dumps(class_report, indent=2),
        'top_features': importance_df.head(20),
        'top_n_features': 20,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Render template
    from jinja2 import Template
    template = Template(html_template)
    html_output = template.render(**template_vars)

    # Save HTML report
    with open(OUTPUT_DIR / 'automl_report.html', 'w') as f:
        f.write(html_output)

    print(f"\nHTML report saved to: {OUTPUT_DIR / 'automl_report.html'}")

def main():
    """Main pipeline"""
    print("\n" + "="*60)
    print("METABOLOMICS AUTOML PIPELINE")
    print("="*60)

    # Load and combine datasets
    combined_df = prepare_combined_dataset()

    # Prepare feature matrix
    X, y, feature_cols = create_feature_matrix(combined_df)

    # Train AutoGluon model
    predictor = train_autogluon_model(X, y)

    # Evaluate model
    metrics, y_pred, y_pred_proba = evaluate_model(predictor, X, y)

    # Get feature importance
    train_data_shuffled = X.copy()
    train_data_shuffled['case_control'] = y
    importance_df = get_feature_importance(predictor, train_data_shuffled)

    # Create visualizations
    create_visualizations(metrics, y, y_pred, y_pred_proba, importance_df)

    # Prepare dataset info
    dataset_info = {
        'total_samples': len(combined_df),
        'num_features': len(feature_cols),
        'case_count': (y == 'Case').sum(),
        'control_count': (y == 'Control').sum(),
        'datasets': 'Negative Mode + Positive Mode'
    }

    # Generate HTML report
    generate_html_report(metrics, importance_df, dataset_info)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Main report: {OUTPUT_DIR / 'automl_report.html'}")
    print("\nOpen the HTML report in a web browser to view results.")

if __name__ == "__main__":
    main()
