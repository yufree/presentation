# Metabolomics AutoML Analysis

## Overview

This project implements an automated machine learning pipeline for predicting case-control status from metabolomics data using **AutoGluon**. The pipeline processes both negative and positive mode LC-MS data and builds an ensemble model for robust predictions.

## 🎯 Results Summary

### Model Performance
- **ROC AUC:** 0.9531 (Excellent)
- **Accuracy:** 88%
- **Precision:** 89% (Case), 87% (Control)
- **Recall:** 85% (Case), 91% (Control)
- **F1-Score:** 87% (Case), 89% (Control)

### Dataset
- **Total Samples:** 2,010 (1,072 Controls, 938 Cases)
- **Features:** 3,164 metabolomics features
- **Data Sources:** 
  - Negative mode LC-MS: 1,005 samples
  - Positive mode LC-MS: 1,005 samples

## 📁 Project Structure

```
automldemo/
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
├── automl_pipeline.py         # Main pipeline script
├── MTBLS28negmzrt.csv        # Negative mode data
├── MTBLS28posmzrt.csv        # Positive mode data
├── index.html                # Results webpage (OPEN THIS!)
├── AUTOML_RESULTS_SUMMARY.md # Detailed results
└── CLAUDE.md                 # Claude Code guidance
```

## 🚀 Quick Start

### Option 1: View Results (Recommended)
Simply open `index.html` in your web browser to view the interactive results dashboard.

### Option 2: Run the Pipeline

```bash
# Build Docker image
docker build -t automl-metabolomics .

# Run the pipeline
docker run --rm -v $(pwd):/app/data automl-metabolomics
```

## 📊 Key Features

1. **Automated Data Processing**
   - Loads and combines negative/positive mode data
   - Extracts case/control labels from sample metadata
   - Handles missing values and data cleaning

2. **AutoML with AutoGluon**
   - Tests multiple algorithms (LightGBM, XGBoost, Neural Networks, etc.)
   - Creates weighted ensemble model
   - Automatic hyperparameter tuning

3. **Feature Importance Analysis**
   - Identifies most predictive metabolomics features
   - Fast variance-based importance scoring

4. **Comprehensive Visualizations**
   - ROC curves
   - Confusion matrices
   - Feature importance plots
   - Interactive charts

5. **HTML Report Generation**
   - Professional, interactive dashboard
   - Embedded visualizations
   - Detailed metrics and insights

## 🔬 Scientific Insights

### Top 5 Most Important Features
1. **187.0073_182.3883** (59.51% importance)
2. **178.0507_46.7570** (6.37% importance)
3. **230.0127_81.9268** (6.25% importance)
4. **212.0020_128.3515** (4.38% importance)
5. **229.9959_73.6495** (2.60% importance)

*Features are labeled as {mz_rt} where mz = mass-to-charge ratio, rt = retention time*

### Model Architecture
- **Best Model:** WeightedEnsemble_L2
- **Component Models:**
  - NeuralNetTorch (57.1% weight)
  - LightGBM (21.4% weight)
  - LightGBMXT (14.3% weight)
  - CatBoost (7.1% weight)

## 💡 Key Findings

1. **Excellent Discriminative Power:** ROC AUC > 0.95 indicates the model can effectively distinguish between case and control samples

2. **Feature Concentration:** A small number of features contribute significantly to prediction accuracy

3. **Ensemble Superiority:** The weighted ensemble outperformed individual models by combining their strengths

4. **Balanced Performance:** Good performance across both classes without significant bias

## 🔧 Technical Details

- **Framework:** AutoGluon 1.1.0
- **Language:** Python 3.9
- **Container:** Docker
- **Training Time:** ~10 minutes
- **Memory Optimized:** Dynamic stacking disabled to prevent OOM errors
- **Validation:** 20% holdout set

## 📖 Usage

### View Results
Open `index.html` in any modern web browser for an interactive dashboard.

### Run Analysis
```bash
# Ensure Docker is installed
docker --version

# Build and run
cd automldemo/
docker build -t automl-metabolomics .
docker run --rm -v $(pwd):/data automl-metabolomics
```

### Access Outputs
All results are saved to the `/app/output/` directory in the Docker container, including:
- Trained model artifacts
- Feature importance CSV
- Metrics JSON
- Visualizations (PNG and HTML)
- Complete HTML report

## 📝 Files Generated

1. **index.html** - Main results dashboard (OPEN THIS!)
2. **AUTOML_RESULTS_SUMMARY.md** - Detailed text summary
3. **automl_pipeline.py** - Complete pipeline source code
4. **Dockerfile** - Container configuration
5. **pipeline_output.log** - Full execution log

## 🎓 References

- AutoGluon Documentation: https://auto.gluon.ai/
- Metabolomics Dataset: MTBLS28
- Docker: https://www.docker.com/

## 📧 Contact

For questions or issues, please refer to the pipeline output logs or AutoGluon documentation.

---

**Note:** The pipeline is fully containerized and can be run on any system with Docker installed, regardless of local Python/R setup.
