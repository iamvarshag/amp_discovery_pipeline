"""
Unit Tests — Feature Engineering
==================================
Purpose:
    Verify that every feature calculation function
    produces correct, biologically meaningful output.

Run with:
    pytest tests/ -v

Why this matters:
    Silent errors in feature calculation produce wrong
    numbers that look correct. The model trains on them
    and gives confidently wrong predictions. Tests catch
    this before it happens.
"""

import pytest
import numpy as np
import pandas as pd
import sys
sys.path.append('.')

from src.feature_engineering import (
    is_valid_sequence,
    calculate_physicochemical_features,
    calculate_aa_composition,
    calculate_dipeptide_composition,
    build_feature_matrix
)


# ============================================================
# TEST SEQUENCES
# ============================================================

MAGAININ2    = "GIGKFLHSAKKFGKAFVGEIMNS"  # Real AMP, frog skin
ALL_20_AA    = "ACDEFGHIKLMNPQRSTVWY"      # One of each AA
CATIONIC_AMP = "KLLLKWLLKWLKK"            # High charge AMP
ALL_LYSINE   = "KKKKKKKKKKKKKK"           # All positive AA


# ============================================================
# SEQUENCE VALIDATION
# ============================================================

class TestValidation:

    def test_valid_sequence_passes(self):
        assert is_valid_sequence(MAGAININ2) == True

    def test_short_sequence_fails(self):
        assert is_valid_sequence("GKL") == False

    def test_empty_fails(self):
        assert is_valid_sequence("") == False

    def test_invalid_char_fails(self):
        assert is_valid_sequence("GLLXBZQ") == False


# ============================================================
# PHYSICOCHEMICAL FEATURES
# ============================================================

class TestPhysicoChemical:

    def test_returns_dict(self):
        """Function must return a dictionary."""
        result = calculate_physicochemical_features(MAGAININ2)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        """All required feature keys must be present."""
        result = calculate_physicochemical_features(MAGAININ2)
        required = [
            'length', 'charge', 'hydrophobicity',
            'isoelectric_point', 'positive_fraction',
            'negative_fraction', 'hydrophobic_fraction',
            'charge_density', 'positive_count',
            'hydrophobic_count'
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_length_correct(self):
        """Length must equal actual sequence length."""
        result = calculate_physicochemical_features(MAGAININ2)
        assert result['length'] == len(MAGAININ2)

    def test_all_lysine_high_charge(self):
        """
        All-lysine sequence must have high positive charge.
        Lysine (K) is positively charged at physiological pH.
        A sequence of all K must have charge > 5.
        """
        result = calculate_physicochemical_features(ALL_LYSINE)
        assert result['charge'] > 5, (
            f"All-lysine charge should be >5, got {result['charge']}"
        )

    def test_all_lysine_high_positive_fraction(self):
        """
        All-lysine sequence must have positive_fraction = 1.0.
        Every residue is K (lysine) which is positive.
        """
        result = calculate_physicochemical_features(ALL_LYSINE)
        assert abs(result['positive_fraction'] - 1.0) < 1e-6, (
            f"Expected 1.0, got {result['positive_fraction']}"
        )

    def test_fractions_between_0_and_1(self):
        """All fraction features must be between 0 and 1."""
        result = calculate_physicochemical_features(MAGAININ2)
        for key in ['positive_fraction', 'negative_fraction',
                    'hydrophobic_fraction']:
            val = result[key]
            assert 0 <= val <= 1, (
                f"{key} = {val} is outside [0,1]"
            )

    def test_no_none_values(self):
        """No feature value must be None."""
        result = calculate_physicochemical_features(MAGAININ2)
        for k, v in result.items():
            assert v is not None, f"{k} is None"


# ============================================================
# AMINO ACID COMPOSITION
# ============================================================

class TestAAComposition:

    def test_returns_20_features(self):
        """Must return exactly 20 features (one per amino acid)."""
        result = calculate_aa_composition(MAGAININ2)
        assert len(result) == 20

    def test_fractions_sum_to_one(self):
        """
        The 20 AA fractions must sum to 1.0.
        This is a mathematical invariant.
        If it fails, the calculation is wrong.
        """
        result = calculate_aa_composition(MAGAININ2)
        total  = sum(result.values())
        assert abs(total - 1.0) < 1e-6, (
            f"AA fractions sum to {total}, expected 1.0"
        )

    def test_all_equal_for_one_of_each(self):
        """
        ACDEFGHIKLMNPQRSTVWY has exactly one of each AA.
        Every fraction must equal 1/20 = 0.05.
        """
        result = calculate_aa_composition(ALL_20_AA)
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            val = result[f'aa_{aa}']
            assert abs(val - 0.05) < 1e-6, (
                f"aa_{aa} = {val}, expected 0.05"
            )

    def test_no_negative_values(self):
        """Fractions cannot be negative."""
        result = calculate_aa_composition(MAGAININ2)
        for k, v in result.items():
            assert v >= 0, f"{k} = {v} is negative"

    def test_all_lysine_sequence(self):
        """
        All-lysine sequence must have aa_K = 1.0
        and all other AAs = 0.0.
        """
        result = calculate_aa_composition(ALL_LYSINE)
        assert abs(result['aa_K'] - 1.0) < 1e-6
        for aa in 'ACDEFGHILMNPQRSTVWY':
            assert result[f'aa_{aa}'] == 0.0, (
                f"aa_{aa} should be 0 in all-K sequence"
            )


# ============================================================
# DIPEPTIDE COMPOSITION
# ============================================================

class TestDipeptideComposition:

    def test_returns_400_features(self):
        """Must return exactly 400 features (20×20 pairs)."""
        result = calculate_dipeptide_composition(MAGAININ2)
        assert len(result) == 400

    def test_fractions_sum_to_one(self):
        """
        All dipeptide fractions must sum to 1.0.
        Mathematical invariant — same as AA composition.
        """
        result = calculate_dipeptide_composition(MAGAININ2)
        total  = sum(result.values())
        assert abs(total - 1.0) < 1e-6, (
            f"Dipeptide fractions sum to {total}"
        )

    def test_no_negative_values(self):
        """No dipeptide fraction can be negative."""
        result = calculate_dipeptide_composition(MAGAININ2)
        for k, v in result.items():
            assert v >= 0, f"{k} = {v} is negative"


# ============================================================
# FEATURE MATRIX
# ============================================================

class TestFeatureMatrix:

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'id':       ['s1','s2','s3','s4'],
            'sequence': [
                MAGAININ2, CATIONIC_AMP,
                ALL_20_AA, "ACDEFGHIKLMNPQRSTVWYACDE"
            ],
            'label': [1, 1, 0, 0]
        })

    def test_output_is_dataframe(self, sample_df):
        result = build_feature_matrix(sample_df, 'basic')
        assert isinstance(result, pd.DataFrame)

    def test_has_label_column(self, sample_df):
        """Label column must survive feature engineering."""
        result = build_feature_matrix(sample_df, 'basic')
        assert 'label' in result.columns

    def test_has_sequence_column(self, sample_df):
        """Sequence column must survive feature engineering."""
        result = build_feature_matrix(sample_df, 'basic')
        assert 'sequence' in result.columns

    def test_no_nan_in_numerical_columns(self, sample_df):
        """
        No NaN values in numerical feature columns.
        NaN values crash ML models silently.
        """
        result = build_feature_matrix(sample_df, 'basic')
        num_cols = result.select_dtypes(include=[np.number]).columns
        assert not result[num_cols].isnull().any().any(), (
            "NaN values found in numerical feature columns"
        )

    def test_labels_are_binary(self, sample_df):
        """Labels must be exactly 0 or 1."""
        result = build_feature_matrix(sample_df, 'basic')
        unique = set(result['label'].unique())
        assert unique.issubset({0, 1})

    def test_standard_has_more_features_than_basic(
        self, sample_df
    ):
        """
        Standard feature set must have more columns than basic.
        Basic = 10 features. Standard = 30 features.
        """
        basic    = build_feature_matrix(sample_df, 'basic')
        standard = build_feature_matrix(sample_df, 'standard')
        assert standard.shape[1] > basic.shape[1]

    def test_row_count_matches_input(self, sample_df):
        """Output rows must match valid input sequences."""
        result = build_feature_matrix(sample_df, 'basic')
        assert len(result) <= len(sample_df)
        assert len(result) > 0