"""
Model Training Module — AMP Discovery Pipeline
===============================================
Purpose:
    Train machine learning models to classify peptides
    as antimicrobial (AMP) or non-antimicrobial (non-AMP).

Biological Meaning:
    The model learns which combinations of physicochemical
    properties and amino acid patterns distinguish AMPs
    from non-AMPs. This knowledge can then be applied to
    predict whether any new peptide sequence is antimicrobial.

Inputs:  data/processed/features.csv
Outputs: models/random_forest.pkl
         models/xgboost.pkl
         models/logistic_regression.pkl
         results/model_comparison.csv
         results/training_summary.txt

Models Used:
    1. Logistic Regression — simple baseline, highly interpretable
    2. Random Forest — ensemble of decision trees, robust
    3. XGBoost — gradient boosting, typically best performance

Why These Models:
    - All are explainable (unlike neural networks)
    - All work well on tabular biological data
    - All can output probability scores (not just yes/no)
    - SHAP values work natively with all three

Why NOT Deep Learning:
    With 1586 sequences, deep learning would overfit severely.
    These classical ML models are the scientifically correct choice
    for this dataset size. This is not a limitation — it is
    scientifically rigorous decision making.

Evaluation:
    Stratified K-Fold Cross Validation (k=5)
    This means we train on 80% and test on 20%, five times,
    rotating which 20% is held out each time.
    Final performance = average across all 5 folds.
    This gives honest, unbiased performance estimates.
"""

import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# PATHS
# ============================================================

FEATURES_PATH = 'data/processed/features.csv'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'


# ============================================================
# DATA LOADING
# ============================================================

def load_features(path: str):
    """
    Load feature matrix and separate into X (features) and y (labels).

    X = everything the model uses to learn
    y = what the model is trying to predict (0 or 1)

    We drop 'sequence' and 'label' from X because:
    - sequence is raw text, not a number
    - label is what we are predicting, cannot use it as input
    """
    logger.info(f"Loading features from {path}...")

    df = pd.read_csv(path)

    # Separate features from labels
    drop_cols = ['sequence', 'label']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values
    y = df['label'].values
    feature_names = feature_cols

    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")
    logger.info(f"Class distribution: {np.bincount(y)}")
    logger.info(f"Number of features: {len(feature_names)}")

    return X, y, feature_names


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_models():
    """
    Define all models to train and compare.

    Each model is wrapped in a Pipeline with StandardScaler.

    Why StandardScaler:
        Our features have very different scales.
        charge ranges from -5 to +10
        molecular_weight ranges from 500 to 10000
        Without scaling, large-scale features dominate.
        StandardScaler converts everything to mean=0, std=1.
        This is critical for Logistic Regression and SVM.
        Random Forest and XGBoost are scale-invariant but
        we scale anyway for consistency.

    Returns:
        Dictionary of model name to sklearn Pipeline
    """
    models = {

        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            ))
        ]),

        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ))
        ]),

        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('model', xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                verbosity=0
            ))
        ]),

    }

    return models


# ============================================================
# CROSS VALIDATION
# ============================================================

def evaluate_model_cv(
    name: str,
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5
) -> dict:
    """
    Evaluate a model using Stratified K-Fold Cross Validation.

    What is Stratified K-Fold:
        Split data into 5 equal parts (folds).
        Each fold has the same class ratio as the full dataset.
        Train on 4 folds, test on 1 fold.
        Repeat 5 times, each time a different fold is the test set.
        Average the results.

        This gives an honest estimate of how the model will
        perform on sequences it has never seen before.

    Why 5 folds:
        Standard in the field. Balances computation vs reliability.
        With 1586 sequences, each test fold has ~317 sequences.
        That is enough for reliable performance estimation.

    Args:
        name: Model name for logging
        pipeline: sklearn Pipeline with scaler and model
        X: Feature matrix
        y: Labels
        n_splits: Number of CV folds
    Returns:
        Dictionary of metric name to mean and std
    """
    logger.info(f"\nEvaluating: {name}")
    logger.info("-" * 40)

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    scoring = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']

    cv_results = cross_validate(
        pipeline,
        X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    results = {
        'Model': name,
        'Accuracy': cv_results['test_accuracy'].mean(),
        'Accuracy_std': cv_results['test_accuracy'].std(),
        'AUC_ROC': cv_results['test_roc_auc'].mean(),
        'AUC_ROC_std': cv_results['test_roc_auc'].std(),
        'Precision': cv_results['test_precision'].mean(),
        'Precision_std': cv_results['test_precision'].std(),
        'Recall': cv_results['test_recall'].mean(),
        'Recall_std': cv_results['test_recall'].std(),
        'F1': cv_results['test_f1'].mean(),
        'F1_std': cv_results['test_f1'].std(),
        'Train_Accuracy': cv_results['train_accuracy'].mean(),
    }

    # Print results
    print(f"\n{name} Results (5-Fold CV):")
    print(f"  Accuracy:  {results['Accuracy']:.4f} ± {results['Accuracy_std']:.4f}")
    print(f"  AUC-ROC:   {results['AUC_ROC']:.4f} ± {results['AUC_ROC_std']:.4f}")
    print(f"  Precision: {results['Precision']:.4f} ± {results['Precision_std']:.4f}")
    print(f"  Recall:    {results['Recall']:.4f} ± {results['Recall_std']:.4f}")
    print(f"  F1 Score:  {results['F1']:.4f} ± {results['F1_std']:.4f}")
    print(f"  Train Acc: {results['Train_Accuracy']:.4f} (check for overfitting)")

    # Overfitting check
    overfit_gap = results['Train_Accuracy'] - results['Accuracy']
    if overfit_gap > 0.1:
        logger.warning(
            f"Possible overfitting detected. "
            f"Train-Test gap: {overfit_gap:.4f}"
        )
    else:
        logger.info(f"No significant overfitting. Gap: {overfit_gap:.4f}")

    return results


# ============================================================
# FINAL MODEL TRAINING
# ============================================================

def train_final_model(
    name: str,
    pipeline,
    X: np.ndarray,
    y: np.ndarray
):
    """
    Train final model on complete dataset.

    After cross validation confirms performance,
    we train on ALL data to get the best possible model
    for deployment in the web app.

    This model is what gets saved and used for predictions.
    """
    logger.info(f"Training final {name} on full dataset...")
    pipeline.fit(X, y)
    logger.info(f"Final {name} trained successfully.")
    return pipeline


# ============================================================
# SAVE MODELS
# ============================================================

def save_model(pipeline, name: str):
    """
    Save trained model to disk using joblib.

    joblib is the standard way to save sklearn models.
    The saved file can be loaded later for predictions
    without retraining.

    Args:
        pipeline: Trained sklearn Pipeline
        name: Model name (used for filename)
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    filename = name.lower().replace(' ', '_') + '.pkl'
    filepath = os.path.join(MODELS_DIR, filename)

    joblib.dump(pipeline, filepath)
    logger.info(f"Model saved: {filepath}")

    return filepath


# ============================================================
# RESULTS SUMMARY
# ============================================================

def print_model_comparison(all_results: list):
    """
    Print a clean comparison table of all models.
    Identifies the best model for deployment.
    """
    results_df = pd.DataFrame(all_results)

    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Model':<25} {'Accuracy':>10} {'AUC-ROC':>10} {'F1':>10}")
    print("-"*70)

    for _, row in results_df.iterrows():
        print(
            f"{row['Model']:<25} "
            f"{row['Accuracy']:>10.4f} "
            f"{row['AUC_ROC']:>10.4f} "
            f"{row['F1']:>10.4f}"
        )

    print("="*70)

    # Find best model by AUC-ROC
    best_idx = results_df['AUC_ROC'].idxmax()
    best_model = results_df.loc[best_idx, 'Model']
    best_auc = results_df.loc[best_idx, 'AUC_ROC']

    print(f"\nBest Model: {best_model} (AUC-ROC: {best_auc:.4f})")
    print("\nNote: AUC-ROC is the primary metric.")
    print("It measures ability to rank AMPs above non-AMPs.")
    print("0.5 = random chance, 1.0 = perfect separation.")

    return results_df, best_model


def save_results(results_df: pd.DataFrame, summary_text: str):
    """Save results to results/ directory."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved: {csv_path}")

    # Save text summary
    txt_path = os.path.join(RESULTS_DIR, 'training_summary.txt')
    with open(txt_path, 'w') as f:
        f.write(summary_text)
    logger.info(f"Summary saved: {txt_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("="*55)
    logger.info("Model Training Module — AMP Discovery Pipeline")
    logger.info("="*55)

    # Load data
    if not os.path.exists(FEATURES_PATH):
        logger.error(
            "Features not found. Run src/feature_engineering.py first."
        )
        return

    X, y, feature_names = load_features(FEATURES_PATH)

    # Get models
    models = get_models()

    # Evaluate all models with cross validation
    logger.info("\nStarting cross-validation evaluation...")
    logger.info("This evaluates each model honestly on unseen data.")

    all_results = []

    for name, pipeline in models.items():
        results = evaluate_model_cv(name, pipeline, X, y, n_splits=5)
        all_results.append(results)

    # Print comparison
    results_df, best_model_name = print_model_comparison(all_results)

    # Train final models on full dataset
    logger.info("\nTraining final models on complete dataset...")

    trained_models = {}
    for name, pipeline in models.items():
        final_model = train_final_model(name, pipeline, X, y)
        save_model(final_model, name)
        trained_models[name] = final_model

    # Save feature names for later use in explainability
    feature_names_path = os.path.join(MODELS_DIR, 'feature_names.pkl')
    joblib.dump(feature_names, feature_names_path)
    logger.info(f"Feature names saved: {feature_names_path}")

    # Save results
    summary = f"""
AMP Discovery Pipeline — Training Summary
==========================================
Dataset: {len(y)} sequences ({np.sum(y==1)} AMPs + {np.sum(y==0)} non-AMPs)
Features: {X.shape[1]}
Validation: 5-Fold Stratified Cross Validation

Best Model: {best_model_name}

All Results:
{results_df[['Model','Accuracy','AUC_ROC','F1']].to_string(index=False)}

Note: All metrics reported as mean across 5 CV folds.
AUC-ROC is primary metric for AMP classification tasks.
"""
    save_results(results_df, summary)

    print("\n" + "="*55)
    print("TRAINING COMPLETE")
    print("="*55)
    print(f"Models saved in: {MODELS_DIR}/")
    print(f"Results saved in: {RESULTS_DIR}/")
    print("\nNext step: run src/explainability.py")


if __name__ == "__main__":
    main()