"""
Feature Engineering Module — AMP Discovery Pipeline
====================================================
Purpose:
    Convert raw peptide sequences into numerical feature vectors
    that machine learning models can process.

Biological Meaning:
    Antimicrobial peptides share physicochemical properties:
    - Positive charge (attracts to negative bacterial membranes)
    - Amphipathicity (one side hydrophobic, one hydrophilic)
    - Moderate hydrophobicity (~50% hydrophobic residues)
    These properties become our features.

Inputs:  data/processed/dataset.csv
Outputs: data/processed/features.csv

Method:
    1. Physicochemical descriptors (interpretable, fast)
    2. Amino acid composition (what % of each amino acid)
    3. Dipeptide composition (pairs of amino acids)

Limitations:
    - Does not capture sequence order explicitly
    - No 3D structural information
    - ESM-2 embeddings would capture more but need GPU
"""

import pandas as pd
import numpy as np
from peptides import Peptide
import logging
import os
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# BIOLOGY-INFORMED CONSTANTS
# ============================================================

# All 20 standard amino acids
AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')

# Hydrophobic amino acids
# These prefer to hide inside membranes away from water
HYDROPHOBIC_AA = set('AVILMFYW')

# Positively charged amino acids at physiological pH
# K=Lysine, R=Arginine, H=Histidine
# AMPs are rich in these — they attract to bacterial membranes
POSITIVE_AA = set('KRH')

# Negatively charged amino acids
# D=Aspartate, E=Glutamate
NEGATIVE_AA = set('DE')

# Paths
PROCESSED_DIR = 'data/processed'


# ============================================================
# VALIDATION
# ============================================================

def is_valid_sequence(sequence: str) -> bool:
    """Check sequence contains only standard amino acids."""
    return (
        len(sequence) >= 5 and
        set(sequence.upper()).issubset(set(AMINO_ACIDS))
    )


# ============================================================
# PHYSICOCHEMICAL FEATURES
# ============================================================

def calculate_physicochemical_features(sequence: str) -> dict:
    """
    Calculate global physicochemical descriptors for a peptide.

    Biological meaning of each feature:

    LENGTH:
        AMPs are typically 10-50 amino acids long.
        Too short = not enough to span membrane.
        Too long = loses selectivity for bacteria.

    CHARGE:
        AMPs have net positive charge (+2 to +9).
        Bacterial membranes are negatively charged.
        Opposite charges attract — AMP binds to bacteria.
        Human cell membranes are neutral — AMP ignores them.
        This is why AMPs are selective.

    HYDROPHOBICITY:
        How much the peptide likes fatty environments vs water.
        AMPs need ~40-60% hydrophobicity to insert into membranes.
        Too low = cannot insert. Too high = toxic to human cells.

    ISOELECTRIC POINT:
        pH at which net charge is zero.
        High pI = peptide is positive at body pH = good for AMP.

    POSITIVE FRACTION:
        Fraction of K, R, H residues.
        Direct measure of cationic character.
        AMPs typically have >20% positive residues.

    HYDROPHOBIC FRACTION:
        Fraction of A, V, I, L, M, F, Y, W residues.
        AMPs typically 40-60%.

    CHARGE DENSITY:
        Charge divided by length.
        Normalized measure of how charged the peptide is
        relative to its size.

    Args:
        sequence: Amino acid sequence string
    Returns:
        Dictionary of feature name to value
    """
    seq = sequence.upper()
    length = len(seq)

    try:
        pep = Peptide(seq)

        charge = pep.charge(pH=7.4, pKscale="Lehninger")
        hydrophobicity = pep.hydrophobicity(scale="KyteDoolittle")
        isoelectric_point = pep.isoelectric_point(pKscale="EMBOSS")

    except Exception:
        charge = 0.0
        hydrophobicity = 0.0
        isoelectric_point = 7.0

    # Manual calculations
    positive_count = sum(1 for aa in seq if aa in POSITIVE_AA)
    negative_count = sum(1 for aa in seq if aa in NEGATIVE_AA)
    hydrophobic_count = sum(1 for aa in seq if aa in HYDROPHOBIC_AA)

    positive_fraction = positive_count / length
    negative_fraction = negative_count / length
    hydrophobic_fraction = hydrophobic_count / length
    charge_density = charge / length if length > 0 else 0

    return {
        'length': length,
        'charge': charge,
        'hydrophobicity': hydrophobicity,
        'isoelectric_point': isoelectric_point,
        'positive_fraction': positive_fraction,
        'negative_fraction': negative_fraction,
        'hydrophobic_fraction': hydrophobic_fraction,
        'charge_density': charge_density,
        'positive_count': positive_count,
        'hydrophobic_count': hydrophobic_count,
    }


# ============================================================
# AMINO ACID COMPOSITION
# ============================================================

def calculate_aa_composition(sequence: str) -> dict:
    """
    Calculate fraction of each amino acid in sequence.

    Biological meaning:
        If AMPs use more Lysine (K) and Arginine (R),
        those features will be higher for AMPs.
        The model learns: high K + high R = likely AMP.

        This gives 20 features — one per amino acid.
        Values are fractions (sum = 1.0).

    Args:
        sequence: Amino acid sequence string
    Returns:
        Dictionary with keys like 'aa_K', 'aa_R' etc.
    """
    seq = sequence.upper()
    length = len(seq)

    composition = {}
    for aa in AMINO_ACIDS:
        composition[f'aa_{aa}'] = seq.count(aa) / length

    return composition


# ============================================================
# DIPEPTIDE COMPOSITION
# ============================================================

def calculate_dipeptide_composition(sequence: str) -> dict:
    """
    Calculate frequency of consecutive amino acid pairs.

    Biological meaning:
        Amino acid ORDER matters, not just composition.
        'KK' (two lysines together) creates a strong positive patch
        that binds strongly to bacterial membranes.
        'KA' has different properties entirely.

        This gives 400 features (20 x 20 pairs).
        Partially captures sequence order information.

    Args:
        sequence: Amino acid sequence string
    Returns:
        Dictionary with keys like 'dp_KK', 'dp_KA' etc.
    """
    seq = sequence.upper()
    total_pairs = len(seq) - 1

    # Initialize all 400 possible dipeptides to zero
    dipeptides = {}
    for aa1 in AMINO_ACIDS:
        for aa2 in AMINO_ACIDS:
            dipeptides[f'dp_{aa1}{aa2}'] = 0.0

    if total_pairs <= 0:
        return dipeptides

    # Count actual pairs
    for i in range(len(seq) - 1):
        pair = f'dp_{seq[i]}{seq[i+1]}'
        if pair in dipeptides:
            dipeptides[pair] += 1

    # Convert to fractions
    for key in dipeptides:
        dipeptides[key] /= total_pairs

    return dipeptides


# ============================================================
# COMPLETE FEATURE PIPELINE
# ============================================================

def build_feature_matrix(
    df: pd.DataFrame,
    feature_set: str = 'standard'
) -> pd.DataFrame:
    """
    Build complete feature matrix for all sequences.

    Feature sets:
        'basic'    — physicochemical only (10 features, fastest)
        'standard' — physicochemical + AA composition (30 features)
        'full'     — all including dipeptides (430 features, slowest)

    For training: use 'standard' first, 'full' for best accuracy.

    Args:
        df: DataFrame with 'sequence' and 'label' columns
        feature_set: Which features to calculate
    Returns:
        Feature DataFrame ready for ML
    """
    logger.info(f"Building feature matrix — set: {feature_set}")
    logger.info(f"Processing {len(df)} sequences...")

    all_features = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        seq = row['sequence']

        if not is_valid_sequence(seq):
            continue

        # Always calculate physicochemical
        features = {}
        features.update(calculate_physicochemical_features(seq))

        # Add amino acid composition
        if feature_set in ['standard', 'full']:
            features.update(calculate_aa_composition(seq))

        # Add dipeptides
        if feature_set == 'full':
            features.update(calculate_dipeptide_composition(seq))

        # Add metadata
        features['sequence'] = seq
        features['label'] = row['label']

        all_features.append(features)

    feature_df = pd.DataFrame(all_features)

    # Remove any rows with NaN
    original = len(feature_df)
    feature_df = feature_df.dropna()
    dropped = original - len(feature_df)

    if dropped > 0:
        logger.warning(f"Dropped {dropped} sequences with NaN features.")

    logger.info(f"Feature matrix shape: {feature_df.shape}")
    logger.info(
        f"Features per sequence: {feature_df.shape[1] - 2}"
    )

    return feature_df


# ============================================================
# BIOLOGICAL SUMMARY
# ============================================================

def print_biological_summary(feature_df: pd.DataFrame):
    """
    Print mean values of key features for AMP vs non-AMP.

    This is your first biological validation.
    AMPs should show:
    - Higher charge
    - Higher positive fraction
    - Different hydrophobicity pattern

    If AMPs and non-AMPs show similar values for all features,
    the features are not informative and the model will fail.
    """
    amp = feature_df[feature_df['label'] == 1]
    non_amp = feature_df[feature_df['label'] == 0]

    print("\n" + "="*55)
    print("BIOLOGICAL FEATURE COMPARISON: AMP vs non-AMP")
    print("="*55)
    print(f"{'Feature':<25} {'AMP':>10} {'non-AMP':>10} {'Difference':>12}")
    print("-"*55)

    key_features = [
        'charge', 'hydrophobicity', 'isoelectric_point',
        'positive_fraction', 'hydrophobic_fraction',
        'charge_density', 'length'
    ]

    for feat in key_features:
        if feat in feature_df.columns:
            amp_mean = amp[feat].mean()
            non_amp_mean = non_amp[feat].mean()
            diff = amp_mean - non_amp_mean
            print(
                f"{feat:<25} {amp_mean:>10.3f} "
                f"{non_amp_mean:>10.3f} {diff:>+12.3f}"
            )

    print("="*55)
    print("Positive difference = AMP has higher value")
    print("Negative difference = AMP has lower value")


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("="*55)
    logger.info("Feature Engineering Module")
    logger.info("="*55)

    # Load dataset
    dataset_path = os.path.join(PROCESSED_DIR, 'dataset.csv')

    if not os.path.exists(dataset_path):
        logger.error(
            "Dataset not found. Run data/download_data.py first."
        )
        return

    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded {len(df)} sequences.")

    # Build features
    feature_df = build_feature_matrix(df, feature_set='standard')

    # Save
    output_path = os.path.join(PROCESSED_DIR, 'features.csv')
    feature_df.to_csv(output_path, index=False)
    logger.info(f"Features saved to: {output_path}")

    # Print biological summary
    print_biological_summary(feature_df)

    logger.info("\nFeature engineering complete.")
    logger.info("Next step: run src/model_training.py")


if __name__ == "__main__":
    main()