"""
Explainability Module — AMP Discovery Pipeline
===============================================
Purpose:
    Explain WHY the model predicts a peptide as antimicrobial
    using SHAP (SHapley Additive exPlanations) values.

Biological Meaning:
    A prediction without explanation is useless to a biologist.
    SHAP tells us:
    - Which features pushed the prediction toward AMP
    - Which features pushed it toward non-AMP
    - How much each feature contributed to the final score

    Example output:
    "This peptide is predicted AMP because:
     charge (+2.3) pushed strongly toward AMP
     length (45 aa) pushed moderately toward AMP
     hydrophobic_fraction (0.32) pushed slightly toward non-AMP"

    This is actionable. A researcher can now modify the peptide
    to increase its AMP probability by targeting specific features.

What is SHAP:
    SHAP is based on game theory (Shapley values).
    It asks: if we remove each feature one by one,
    how much does the prediction change?
    Features that change the prediction a lot get high SHAP values.
    Features that change it little get low SHAP values.

    SHAP values are:
    - Positive → pushed prediction toward AMP (label=1)
    - Negative → pushed prediction toward non-AMP (label=0)

Inputs:  data/processed/features.csv
         models/xgboost.pkl
         models/feature_names.pkl
Outputs: results/shap_summary.png
         results/shap_bar.png
         results/feature_importance.csv
         results/shap_values.npy
"""

import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import os
import logging
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
# DATA AND MODEL LOADING
# ============================================================

def load_data_and_model():
    """
    Load the feature matrix and trained XGBoost model.

    We use XGBoost for SHAP because:
    1. It is the best performing model
    2. XGBoost has native SHAP support (TreeExplainer)
    3. TreeExplainer is exact and fast for tree-based models
    """
    logger.info("Loading features and model...")

    # Load features
    df = pd.read_csv(FEATURES_PATH)
    drop_cols = ['sequence', 'label']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values
    y = df['label'].values
    sequences = df['sequence'].values

    # Load trained XGBoost model
    model_path = os.path.join(MODELS_DIR, 'xgboost.pkl')
    pipeline = joblib.load(model_path)

    # Load feature names
    feature_names_path = os.path.join(MODELS_DIR, 'feature_names.pkl')
    feature_names = joblib.load(feature_names_path)

    logger.info(f"Loaded {len(X)} sequences, {len(feature_names)} features")

    return X, y, sequences, pipeline, feature_names


# ============================================================
# SHAP COMPUTATION
# ============================================================

def compute_shap_values(X, pipeline, feature_names):
    """
    Compute SHAP values for all sequences.

    We use TreeExplainer specifically for XGBoost/Random Forest.
    TreeExplainer is:
    - Exact (not approximate)
    - Fast for tree-based models
    - Guaranteed to be consistent

    The pipeline has a scaler before XGBoost.
    We need to transform X through the scaler first,
    then pass to XGBoost's SHAP explainer.

    Returns:
        shap_values: array of shape (n_samples, n_features)
        X_scaled: scaled feature matrix
    """
    logger.info("Computing SHAP values...")
    logger.info("This may take 1-2 minutes for 1586 sequences...")

    # Extract scaler and model from pipeline
    scaler = pipeline.named_steps['scaler']
    model = pipeline.named_steps['model']

    # Scale the features
    X_scaled = scaler.transform(X)

    # Create SHAP explainer for XGBoost
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_scaled)

    logger.info(f"SHAP values computed. Shape: {shap_values.shape}")
    logger.info(
        f"Mean absolute SHAP: {np.abs(shap_values).mean():.4f}"
    )

    return shap_values, X_scaled, explainer


# ============================================================
# FEATURE IMPORTANCE FROM SHAP
# ============================================================

def compute_feature_importance(
    shap_values: np.ndarray,
    feature_names: list
) -> pd.DataFrame:
    """
    Rank features by their mean absolute SHAP value.

    Mean |SHAP| = average impact of this feature on predictions.
    Higher = more important for the model's decisions.

    This is more reliable than built-in feature importance
    because SHAP accounts for feature interactions.

    Biological interpretation:
    If 'charge' has the highest mean |SHAP|, it means
    charge is the most important property for distinguishing
    AMPs from non-AMPs in our dataset.
    This aligns with known AMP biology — charge is critical.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
    })

    importance_df = importance_df.sort_values(
        'mean_abs_shap',
        ascending=False
    ).reset_index(drop=True)

    importance_df['rank'] = importance_df.index + 1

    return importance_df


# ============================================================
# VISUALIZATIONS
# ============================================================

def plot_shap_bar(importance_df: pd.DataFrame, top_n: int = 15):
    """
    Bar plot of top N most important features by mean |SHAP|.

    This is the clearest visualization for:
    - README figures
    - Paper figures
    - Presentations
    - Web app display

    Biological meaning on the plot:
    Each bar shows how much that feature influences predictions
    on average across all 1586 sequences.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    top_features = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ['#FF6B6B' if i < 5 else '#4ECDC4'
              for i in range(len(top_features))]

    bars = ax.barh(
        range(len(top_features)),
        top_features['mean_abs_shap'],
        color=colors,
        edgecolor='white',
        linewidth=0.5
    )

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'], fontsize=11)
    ax.invert_yaxis()

    ax.set_xlabel('Mean |SHAP Value| (Impact on Prediction)', fontsize=12)
    ax.set_title(
        f'Top {top_n} Features by SHAP Importance\n'
        f'AMP Discovery Pipeline — XGBoost Model',
        fontsize=13,
        fontweight='bold'
    )

    # Add value labels on bars
    for i, (bar, val) in enumerate(
        zip(bars, top_features['mean_abs_shap'])
    ):
        ax.text(
            val + 0.001,
            bar.get_y() + bar.get_height()/2,
            f'{val:.4f}',
            va='center',
            fontsize=9
        )

    ax.text(
        0.98, 0.02,
        'Red = Top 5 most important',
        transform=ax.transAxes,
        ha='right',
        fontsize=9,
        color='#FF6B6B'
    )

    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, 'shap_bar.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"SHAP bar plot saved: {output_path}")


def plot_shap_summary(
    shap_values: np.ndarray,
    X_scaled: np.ndarray,
    feature_names: list,
    top_n: int = 15
):
    """
    SHAP beeswarm summary plot.

    This is the most informative SHAP visualization.
    Each dot is one sequence.
    X axis = SHAP value (positive = toward AMP)
    Color = feature value (red = high, blue = low)

    How to read it:
    If 'charge' shows red dots on the right side,
    it means HIGH charge → pushes toward AMP prediction.
    This is exactly what biology predicts.

    This plot validates that the model learned real biology,
    not statistical artifacts.
    """
    # Select top N features for clarity
    importance = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(importance)[-top_n:][::-1]

    shap_top = shap_values[:, top_indices]
    X_top = X_scaled[:, top_indices]
    names_top = [feature_names[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(10, 8))

    shap.summary_plot(
        shap_top,
        X_top,
        feature_names=names_top,
        show=False,
        plot_type='dot',
        max_display=top_n,
        alpha=0.6
    )

    plt.title(
        'SHAP Summary Plot — Feature Impact on AMP Prediction\n'
        'Right = pushes toward AMP | Left = pushes toward non-AMP',
        fontsize=12,
        fontweight='bold'
    )

    output_path = os.path.join(RESULTS_DIR, 'shap_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"SHAP summary plot saved: {output_path}")


def plot_biological_validation(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list,
    y: np.ndarray
):
    """
    Plot showing SHAP values for key biological features
    separately for AMP and non-AMP groups.

    This is your biological validation figure.
    It shows the model learned genuine biology:
    - AMPs have positive SHAP for charge
    - non-AMPs have negative SHAP for charge
    This matches known AMP science.
    """
    key_features = [
        'charge',
        'isoelectric_point',
        'positive_fraction',
        'hydrophobicity',
        'length',
        'charge_density'
    ]

    available = [f for f in key_features if f in feature_names]
    if not available:
        logger.warning("Key biological features not found in feature names.")
        return

    n_features = len(available)
    fig, axes = plt.subplots(
        1, n_features,
        figsize=(3 * n_features, 5)
    )

    if n_features == 1:
        axes = [axes]

    amp_idx = np.where(y == 1)[0]
    non_amp_idx = np.where(y == 0)[0]

    for ax, feat in zip(axes, available):
        if feat not in feature_names:
            continue

        feat_idx = list(feature_names).index(feat)
        shap_col = shap_values[:, feat_idx]

        ax.violinplot(
            [shap_col[amp_idx], shap_col[non_amp_idx]],
            positions=[0, 1],
            showmeans=True
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['AMP', 'non-AMP'], fontsize=10)
        ax.set_title(feat, fontsize=10, fontweight='bold')
        ax.set_ylabel('SHAP Value', fontsize=9)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    plt.suptitle(
        'SHAP Values by Class — Biological Validation\n'
        'Positive SHAP = pushes toward AMP prediction',
        fontsize=12,
        fontweight='bold'
    )

    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, 'shap_biological_validation.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Biological validation plot saved: {output_path}")


# ============================================================
# SINGLE SEQUENCE EXPLANATION
# ============================================================

def explain_single_sequence(
    sequence: str,
    pipeline,
    feature_names: list,
    explainer,
    scaler
) -> dict:
    """
    Explain prediction for a single new peptide sequence.

    This is what powers the web app.
    User pastes a sequence → we return:
    1. AMP probability
    2. Top features driving the prediction
    3. Biological interpretation

    Args:
        sequence: Amino acid sequence string
        pipeline: Trained XGBoost pipeline
        feature_names: List of feature names
        explainer: SHAP TreeExplainer
        scaler: Fitted StandardScaler
    Returns:
        Dictionary with prediction and explanation
    """
    # Import feature engineering functions
    import sys
    sys.path.append('.')
    from src.feature_engineering import (
        calculate_physicochemical_features,
        calculate_aa_composition
    )

    seq = sequence.upper().strip()

    # Calculate features
    features = {}
    features.update(calculate_physicochemical_features(seq))
    features.update(calculate_aa_composition(seq))

    # Build feature vector in correct order
    feature_vector = np.array(
        [features.get(f, 0.0) for f in feature_names]
    ).reshape(1, -1)

    # Scale
    feature_vector_scaled = scaler.transform(feature_vector)

    # Predict
    model = pipeline.named_steps['model']
    probability = model.predict_proba(feature_vector_scaled)[0][1]
    prediction = 'AMP' if probability >= 0.5 else 'non-AMP'

    # SHAP explanation
    shap_vals = explainer.shap_values(feature_vector_scaled)[0]

    # Top contributing features
    feature_contributions = list(zip(feature_names, shap_vals))
    feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    top_positive = [
        (f, v) for f, v in feature_contributions if v > 0
    ][:5]

    top_negative = [
        (f, v) for f, v in feature_contributions if v < 0
    ][:5]

    return {
        'sequence': seq,
        'prediction': prediction,
        'amp_probability': float(probability),
        'top_amp_drivers': top_positive,
        'top_non_amp_drivers': top_negative,
        'all_shap_values': dict(zip(feature_names, shap_vals)),
        'raw_features': features
    }


def print_single_explanation(result: dict):
    """Print a human-readable explanation of a single prediction."""

    print("\n" + "="*55)
    print("SINGLE SEQUENCE EXPLANATION")
    print("="*55)
    print(f"Sequence:    {result['sequence']}")
    print(f"Prediction:  {result['prediction']}")
    print(f"Probability: {result['amp_probability']:.4f} "
          f"({result['amp_probability']*100:.1f}% AMP)")
    print("\nTop features pushing toward AMP:")
    for feat, val in result['top_amp_drivers']:
        print(f"  + {feat:<25} SHAP: {val:+.4f}")
    print("\nTop features pushing toward non-AMP:")
    for feat, val in result['top_non_amp_drivers']:
        print(f"  - {feat:<25} SHAP: {val:+.4f}")
    print("="*55)


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("="*55)
    logger.info("Explainability Module — SHAP Analysis")
    logger.info("="*55)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data and model
    X, y, sequences, pipeline, feature_names = load_data_and_model()

    # Compute SHAP values
    shap_values, X_scaled, explainer = compute_shap_values(
        X, pipeline, feature_names
    )

    # Save SHAP values for later use
    np.save(
        os.path.join(RESULTS_DIR, 'shap_values.npy'),
        shap_values
    )

    # Compute feature importance
    importance_df = compute_feature_importance(shap_values, feature_names)

    # Save importance
    importance_path = os.path.join(RESULTS_DIR, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Feature importance saved: {importance_path}")

    # Print top 10 features
    print("\n" + "="*55)
    print("TOP 10 MOST IMPORTANT FEATURES (by SHAP)")
    print("="*55)
    print(f"{'Rank':<6} {'Feature':<25} {'Mean |SHAP|':>12}")
    print("-"*55)
    for _, row in importance_df.head(10).iterrows():
        print(
            f"{int(row['rank']):<6} "
            f"{row['feature']:<25} "
            f"{row['mean_abs_shap']:>12.4f}"
        )
    print("="*55)

    # Generate plots
    logger.info("\nGenerating SHAP visualizations...")
    plot_shap_bar(importance_df, top_n=15)
    plot_shap_summary(shap_values, X_scaled, feature_names, top_n=15)
    plot_biological_validation(shap_values, X, feature_names, y)

    # Test on real known AMPs
    logger.info("\nTesting explanation on known AMPs...")

    scaler = pipeline.named_steps['scaler']

    test_sequences = [
        ("GIGKFLHSAKKFGKAFVGEIMNS", "Magainin-2 (known AMP, frog skin)"),
        ("ACDEFGHIKLMNPQRSTVWYACDE", "Control non-AMP sequence"),
        ("KLLLKWLLKWLKK", "Synthetic cationic AMP"),
    ]

    for seq, description in test_sequences:
        print(f"\nTesting: {description}")
        result = explain_single_sequence(
            seq, pipeline, feature_names, explainer, scaler
        )
        print_single_explanation(result)

    print("\n" + "="*55)
    print("EXPLAINABILITY COMPLETE")
    print("="*55)
    print("Files saved in results/:")
    print("  - shap_bar.png")
    print("  - shap_summary.png")
    print("  - shap_biological_validation.png")
    print("  - feature_importance.csv")
    print("  - shap_values.npy")
    print("\nNext step: build the Streamlit web app")


if __name__ == "__main__":
    main()