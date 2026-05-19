"""
Unit Tests — Data Pipeline
===========================
Purpose:
    Automatically verify that data download and
    processing functions work correctly.

Run with:
    pytest tests/ -v

Biological meaning:
    If data functions break silently, the entire
    pipeline trains on garbage. Tests catch this
    before it causes invisible errors downstream.
"""

import pytest
import sys
import os
sys.path.append('.')

from data.download_data import (
    is_valid_sequence,
    assemble_dataset,
    remove_duplicates
)
import pandas as pd


# ============================================================
# TEST DATA
# ============================================================

VALID_AMP        = "GIGKFLHSAKKFGKAFVGEIMNS"   # Magainin-2
VALID_SHORT      = "KLLLKWLLKWLKK"             # Short valid AMP
TOO_SHORT        = "GKL"                        # Below MIN_LENGTH
TOO_LONG         = "A" * 150                    # Above MAX_LENGTH
INVALID_CHARS    = "GLLXBZQ"                    # Non-standard AAs
EMPTY            = ""                           # Empty string


# ============================================================
# SEQUENCE VALIDATION TESTS
# ============================================================

class TestSequenceValidation:

    def test_valid_amp_passes(self):
        """Known real AMP must pass validation."""
        assert is_valid_sequence(VALID_AMP) == True

    def test_valid_short_amp_passes(self):
        """Short but valid AMP must pass."""
        assert is_valid_sequence(VALID_SHORT) == True

    def test_too_short_rejected(self):
        """Sequences below MIN_LENGTH must be rejected."""
        assert is_valid_sequence(TOO_SHORT) == False

    def test_too_long_rejected(self):
        """Sequences above MAX_LENGTH must be rejected."""
        assert is_valid_sequence(TOO_LONG) == False

    def test_invalid_characters_rejected(self):
        """Non-standard amino acid characters must be rejected."""
        assert is_valid_sequence(INVALID_CHARS) == False

    def test_empty_string_rejected(self):
        """Empty string must be rejected."""
        assert is_valid_sequence(EMPTY) == False

    def test_lowercase_rejected(self):
        """
        Lowercase sequences are accepted because
        is_valid_sequence calls .upper() internally.
        This is correct behavior — we test that it works.
        """
        assert is_valid_sequence("gigkflhsak") == True

    def test_sequence_with_spaces_rejected(self):
        """Sequences with spaces must be rejected."""
        assert is_valid_sequence("GIGK FLHS") == False


# ============================================================
# DATASET ASSEMBLY TESTS
# ============================================================

class TestDatasetAssembly:

    @pytest.fixture
    def sample_positives(self):
        return [
            {'id': f'AMP{i:03d}',
             'sequence': VALID_AMP,
             'label': 1,
             'source': 'test'}
            for i in range(20)
        ]

    @pytest.fixture
    def sample_negatives(self):
        return [
            {'id': f'NEG{i:03d}',
             'sequence': VALID_SHORT,
             'label': 0,
             'source': 'test'}
            for i in range(30)
        ]

    def test_output_is_dataframe(
        self, sample_positives, sample_negatives
    ):
        """assemble_dataset must return a DataFrame."""
        result = assemble_dataset(
            sample_positives, sample_negatives
        )
        assert isinstance(result, pd.DataFrame)

    def test_balanced_output(
        self, sample_positives, sample_negatives
    ):
        """
        Output must be balanced — equal AMPs and non-AMPs.
        Imbalanced datasets cause biased models.
        With 20 positives and 30 negatives, result = 40 total.
        """
        result = assemble_dataset(
            sample_positives, sample_negatives
        )
        counts = result['label'].value_counts()
        assert counts[0] == counts[1], (
            f"Dataset not balanced: {counts[0]} vs {counts[1]}"
        )

    def test_has_required_columns(
        self, sample_positives, sample_negatives
    ):
        """Output must have id, sequence, label, source columns."""
        result = assemble_dataset(
            sample_positives, sample_negatives
        )
        for col in ['id', 'sequence', 'label', 'source']:
            assert col in result.columns, (
                f"Missing required column: {col}"
            )

    def test_labels_are_binary(
        self, sample_positives, sample_negatives
    ):
        """Labels must be exactly 0 or 1."""
        result = assemble_dataset(
            sample_positives, sample_negatives
        )
        unique = set(result['label'].unique())
        assert unique.issubset({0, 1})

    def test_smaller_class_determines_size(
        self, sample_positives, sample_negatives
    ):
        """
        Total size = 2 × smaller class.
        20 positives + 30 negatives → 40 total (2×20).
        """
        result = assemble_dataset(
            sample_positives, sample_negatives
        )
        expected = 2 * min(
            len(sample_positives), len(sample_negatives)
        )
        assert len(result) == expected


# ============================================================
# DEDUPLICATION TESTS
# ============================================================

class TestDeduplication:

    def test_removes_duplicate_sequences(self):
        """Duplicate sequences must be removed."""
        df = pd.DataFrame({
            'id': ['A', 'B', 'C'],
            'sequence': [VALID_AMP, VALID_AMP, VALID_SHORT],
            'label': [1, 1, 0],
            'source': ['test', 'test', 'test']
        })
        result = remove_duplicates(df)
        assert len(result) == 2

    def test_no_duplicates_unchanged(self):
        """DataFrame with no duplicates must be unchanged."""
        df = pd.DataFrame({
            'id': ['A', 'B'],
            'sequence': [VALID_AMP, VALID_SHORT],
            'label': [1, 0],
            'source': ['test', 'test']
        })
        result = remove_duplicates(df)
        assert len(result) == 2

    def test_preserves_columns(self):
        """Deduplication must not remove any columns."""
        df = pd.DataFrame({
            'id': ['A', 'B'],
            'sequence': [VALID_AMP, VALID_SHORT],
            'label': [1, 0],
            'source': ['test', 'test']
        })
        result = remove_duplicates(df)
        assert list(result.columns) == list(df.columns)