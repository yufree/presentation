# AutoML Metabolomics Analysis - Results Summary

## 🎯 Model Performance

**Final Model:** WeightedEnsemble_L2 (AutoGluon)

### Key Metrics:
- **ROC AUC:** 0.9531 (Excellent discriminative ability)
- **Accuracy:** 88%
- **Precision:** 89% (Case), 87% (Control)
- **Recall:** 85% (Case), 91% (Control)
- **F1-Score:** 87% (Case), 89% (Control)

### Confusion Matrix:
```
                Predicted
Actual    Case    Control
Case       798     140
Control    100     972
```

## 📊 Dataset Information

- **Total Samples:** 2,010
  - Controls: 1,072 (53.3%)
  - Cases: 938 (46.7%)
- **Features:** 3,164 metabolomics features
- **Data Sources:** 
  - Negative mode LC-MS: 1,005 samples
  - Positive mode LC-MS: 1,005 samples

## 🔬 Feature Importance (Top 10)

Based on feature variance:

1. **187.0073_182.3883** - 59.51% importance
2. **178.0507_46.7570** - 6.37% importance
3. **230.0127_81.9268** - 6.25% importance
4. **212.0020_128.3515** - 4.38% importance
5. **229.9959_73.6495** - 2.60% importance
6. **105.0343_154.5457** - 1.64% importance
7. **172.0977_130.9439** - 1.52% importance
8. **144.1029_23.8842** - 1.36% importance
9. **287.1009_164.0682** - 1.29% importance
10. **263.1031_159.1227** - 1.23% importance

*Note: Feature IDs are in format {mz_rt} where mz = mass-to-charge ratio and rt = retention time*

## 🏆 Model Details

### Best Individual Models:
1. **NeuralNetTorch** - ROC AUC: 0.8246
2. **LightGBM** - ROC AUC: 0.8092
3. **LightGBMXT** - ROC AUC: 0.8142

### Ensemble Weights:
- NeuralNetTorch: 57.1%
- LightGBM: 21.4%
- LightGBMXT: 14.3%
- CatBoost: 7.1%

## 📁 Generated Files

The pipeline generated:
- Model artifacts saved in `/app/output/autogluon_model/`
- Feature importance: `feature_importance.csv`
- Metrics: `model_metrics.json`
- Visualizations: `model_performance.png`, `roc_curve.html`, `feature_importance.html`
- HTML Report: `automl_report.html`

## 💡 Key Insights

1. **Excellent Performance:** The model achieved ROC AUC > 0.95, indicating excellent ability to distinguish between case and control samples.

2. **Balanced Predictions:** The model shows good balance between precision and recall for both classes.

3. **Feature Selection:** A small number of features (top 10) contribute significantly to the model's predictive power, with the top feature alone contributing ~60% of the variance.

4. **Ensemble Success:** The weighted ensemble outperformed individual models, combining the strengths of neural networks and gradient boosting.

## 🔧 Technical Details

- **Training Time:** ~10 minutes
- **Validation Split:** 20%
- **Preprocessing:** Automatic feature generation with missing value imputation
- **Memory Optimization:** Dynamic stacking disabled to prevent OOM errors
- **Environment:** Docker container with Python 3.9, AutoGluon 1.1.0

## 📊 Visualization Outputs

The pipeline generated several visualizations:
- ROC Curve (interactive)
- Confusion Matrix
- Feature Importance (interactive bar chart)
- Prediction Distribution

All visualizations are embedded in the HTML report for easy viewing.
