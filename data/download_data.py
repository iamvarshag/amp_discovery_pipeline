"""
Data Download Module — AMP Discovery Pipeline
==============================================
Purpose:
    Downloads antimicrobial peptide sequences (positive class)
    from multiple databases and non-AMP sequences (negative class)
    from UniProt database.

Biological Meaning:
    Positive class = confirmed antimicrobial peptides
    Negative class = proteins with NO antimicrobial activity
    Multiple sources = larger, more diverse, more robust dataset

Inputs:  None
Outputs: data/processed/dataset.csv

Sources:
    Positive: UniProt KW-0929, UniProt KW-0645, DRAMP database
    Negative: UniProt SwissProt short non-antimicrobial sequences

Limitations:
    - Unreviewed UniProt entries are less curated
    - DRAMP may have some overlap with UniProt
    - We deduplicate by sequence to handle overlaps
"""

import os
import requests
import pandas as pd
from Bio import SeqIO
from io import StringIO
import time
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

MIN_LENGTH = 10
MAX_LENGTH = 100
STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')

RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'


# ============================================================
# UTILITIES
# ============================================================

def ensure_directories():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    logger.info("Directories ready.")


def is_valid_sequence(sequence: str) -> bool:
    """
    Validates peptide sequence.
    Removes sequences with unknown characters like X, B, Z
    which break physicochemical feature calculation.
    """
    return (
        MIN_LENGTH <= len(sequence) <= MAX_LENGTH and
        set(sequence.upper()).issubset(STANDARD_AA)
    )


def download_uniprot_batch(
    query: str,
    size: int = 500,
    label: int = 1,
    source_name: str = 'UniProt'
) -> list:
    """
    Generic UniProt downloader.
    Reusable for any query — positive or negative class.

    Args:
        query: UniProt search query string
        size: Number of sequences to request
        label: 1 for AMP, 0 for non-AMP
        source_name: Label for tracking data source
    Returns:
        List of sequence dictionaries
    """
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        'query': query,
        'format': 'fasta',
        'size': min(size, 500),
    }

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()

        sequences = []
        fasta_content = StringIO(response.text)

        for record in SeqIO.parse(fasta_content, "fasta"):
            seq = str(record.seq).upper()
            if is_valid_sequence(seq):
                sequences.append({
                    'id': str(record.id),
                    'sequence': seq,
                    'label': label,
                    'source': source_name
                })

        logger.info(
            f"{source_name}: downloaded {len(sequences)} sequences."
        )
        return sequences

    except requests.exceptions.RequestException as e:
        logger.warning(f"{source_name} download failed: {e}")
        return []


# ============================================================
# POSITIVE CLASS — Multiple AMP Sources
# ============================================================

def download_amps_uniprot_reviewed() -> list:
    """
    Source 1: UniProt SwissProt manually reviewed AMPs.
    KW-0929 = Antimicrobial keyword.
    Highest quality — every entry verified by human curators.
    """
    logger.info("Source 1: UniProt reviewed AMPs (KW-0929)...")
    return download_uniprot_batch(
        query='reviewed:true AND keyword:KW-0929 AND length:[10 TO 100]',
        size=500,
        label=1,
        source_name='UniProt_reviewed_AMP'
    )


def download_amps_uniprot_antibiotic() -> list:
    """
    Source 2: UniProt sequences with antibiotic activity annotation.
    KW-0045 = Antibiotic keyword.
    Captures AMPs annotated differently from KW-0929.
    Adds diversity to positive class.
    """
    logger.info("Source 2: UniProt antibiotic sequences (KW-0045)...")
    return download_uniprot_batch(
        query='reviewed:true AND keyword:KW-0045 AND length:[10 TO 100]',
        size=500,
        label=1,
        source_name='UniProt_antibiotic'
    )


def download_amps_uniprot_unreviewed() -> list:
    """
    Source 3: UniProt TrEMBL unreviewed AMPs.
    Lower quality than SwissProt but much larger volume.
    Adds sequence diversity — different organisms, different families.
    We accept this tradeoff to reach 2000+ positive sequences.
    """
    logger.info("Source 3: UniProt unreviewed AMPs (TrEMBL)...")
    return download_uniprot_batch(
        query=(
            'reviewed:false '
            'AND keyword:KW-0929 '
            'AND length:[10 TO 100] '
            'AND existence:1'
        ),
        size=500,
        label=1,
        source_name='UniProt_unreviewed_AMP'
    )


def download_amps_uniprot_defensins() -> list:
    """
    Source 4: Defensins — major AMP family.
    Defensins are the most studied class of AMPs.
    Found in humans, animals, plants, fungi.
    Specifically targeting this family adds depth.
    """
    logger.info("Source 4: Defensins family AMPs...")
    return download_uniprot_batch(
        query=(
            'reviewed:true '
            'AND family:"defensin" '
            'AND length:[10 TO 100]'
        ),
        size=300,
        label=1,
        source_name='UniProt_defensins'
    )


def download_amps_uniprot_cathelicidins() -> list:
    """
    Source 5: Cathelicidins — another major AMP family.
    LL-37 (human cathelicidin) is one of the most studied AMPs.
    Important for immune defense in mammals.
    """
    logger.info("Source 5: Cathelicidins family AMPs...")
    return download_uniprot_batch(
        query=(
            'reviewed:true '
            'AND family:"cathelicidin" '
            'AND length:[10 TO 100]'
        ),
        size=300,
        label=1,
        source_name='UniProt_cathelicidins'
    )


def download_amps_bacteriocins() -> list:
    """
    Source 6: Bacteriocins — AMPs produced by bacteria.
    Bacteria produce AMPs to kill competing bacteria.
    This adds evolutionary diversity to our dataset.
    """
    logger.info("Source 6: Bacteriocins...")
    return download_uniprot_batch(
        query=(
            'reviewed:true '
            'AND keyword:KW-0078 '
            'AND length:[10 TO 100]'
        ),
        size=300,
        label=1,
        source_name='UniProt_bacteriocins'
    )


def collect_all_positives() -> list:
    """
    Collect AMPs from all sources.
    Deduplicates by sequence to remove overlaps between sources.
    """
    logger.info("="*50)
    logger.info("Collecting positive class (AMPs) from all sources")
    logger.info("="*50)

    all_positives = []

    # Download from each source with delay between requests
    sources = [
        download_amps_uniprot_reviewed,
        download_amps_uniprot_antibiotic,
        download_amps_uniprot_unreviewed,
        download_amps_uniprot_defensins,
        download_amps_uniprot_cathelicidins,
        download_amps_bacteriocins,
    ]

    for source_fn in sources:
        results = source_fn()
        all_positives.extend(results)
        time.sleep(1)  # Be polite to UniProt servers

    # Deduplicate by sequence
    seen_sequences = set()
    unique_positives = []

    for item in all_positives:
        seq = item['sequence']
        if seq not in seen_sequences:
            seen_sequences.add(seq)
            unique_positives.append(item)

    logger.info(f"\nTotal unique AMP sequences: {len(unique_positives)}")

    # Show source breakdown
    source_df = pd.DataFrame(unique_positives)
    if len(source_df) > 0:
        logger.info("Source breakdown:")
        print(source_df['source'].value_counts().to_string())

    return unique_positives


# ============================================================
# NEGATIVE CLASS — Non-AMP sequences
# ============================================================

def download_negatives(n_needed: int) -> list:
    """
    Download non-AMP sequences to match positive class size.

    Strategy:
        Query multiple categories of short non-antimicrobial proteins.
        Hormones, enzymes, structural proteins — all non-antimicrobial.
        This creates biologically meaningful negatives.

    Why this matters:
        Random sequences would be too easy to distinguish from AMPs.
        Real non-AMP proteins are a harder, more realistic challenge.
    """
    logger.info("="*50)
    logger.info(f"Collecting negative class ({n_needed} non-AMPs needed)")
    logger.info("="*50)

    all_negatives = []

    negative_queries = [
        (
            'reviewed:true '
            'AND length:[10 TO 100] '
            'NOT keyword:KW-0929 '
            'NOT keyword:KW-0045 '
            'NOT keyword:KW-0078 '
            'AND keyword:KW-0134',  # Hormone
            'UniProt_hormones'
        ),
        (
            'reviewed:true '
            'AND length:[10 TO 100] '
            'NOT keyword:KW-0929 '
            'NOT keyword:KW-0045 '
            'AND keyword:KW-0547',  # Neuropeptide
            'UniProt_neuropeptides'
        ),
        (
            'reviewed:true '
            'AND length:[10 TO 100] '
            'NOT keyword:KW-0929 '
            'NOT keyword:KW-0045 '
            'AND keyword:KW-0167',  # Transcription regulation
            'UniProt_transcription'
        ),
        (
            'reviewed:true '
            'AND length:[10 TO 100] '
            'NOT keyword:KW-0929 '
            'NOT keyword:KW-0045 '
            'AND keyword:KW-0349',  # Growth factor
            'UniProt_growth_factors'
        ),
    ]

    for query, source_name in negative_queries:
        results = download_uniprot_batch(
            query=query,
            size=500,
            label=0,
            source_name=source_name
        )
        all_negatives.extend(results)
        time.sleep(1)

    # Deduplicate
    seen = set()
    unique_negatives = []
    for item in all_negatives:
        seq = item['sequence']
        if seq not in seen:
            seen.add(seq)
            unique_negatives.append(item)

    logger.info(f"Total unique non-AMP sequences: {len(unique_negatives)}")
    return unique_negatives


# ============================================================
# DATASET ASSEMBLY
# ============================================================

def assemble_dataset(
    positives: list,
    negatives: list
) -> pd.DataFrame:
    """
    Combine and balance positive and negative sequences.

    Balancing strategy:
        Match to the smaller class size.
        This prevents class imbalance bias in ML training.

    If positives > negatives:
        We undersample positives to match negatives.
    If negatives > positives:
        We undersample negatives to match positives.
    """
    pos_df = pd.DataFrame(positives)
    neg_df = pd.DataFrame(negatives)

    min_size = min(len(pos_df), len(neg_df))

    pos_df = pos_df.sample(n=min_size, random_state=42)
    neg_df = neg_df.sample(n=min_size, random_state=42)

    dataset = pd.concat([pos_df, neg_df], ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(
        f"\nFinal balanced dataset: "
        f"{min_size} AMPs + {min_size} non-AMPs = {len(dataset)} total"
    )

    return dataset


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identical sequences — prevents data leakage."""
    before = len(df)
    df = df.drop_duplicates(subset=['sequence'])
    removed = before - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} cross-source duplicates.")
    return df


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("="*55)
    logger.info("AMP Discovery Pipeline — Enhanced Data Download")
    logger.info("="*55)

    ensure_directories()

    # Collect all positives from 6 sources
    positives = collect_all_positives()

    if not positives:
        logger.error("No positive sequences downloaded.")
        return

    logger.info(f"\nTotal AMPs collected: {len(positives)}")
    time.sleep(2)

    # Collect negatives to match
    negatives = download_negatives(n_needed=len(positives))

    if not negatives:
        logger.error("No negative sequences downloaded.")
        return

    # Assemble balanced dataset
    dataset = assemble_dataset(positives, negatives)
    dataset = remove_duplicates(dataset)

    # Save
    output_path = os.path.join(PROCESSED_DIR, 'dataset.csv')
    dataset.to_csv(output_path, index=False)

    logger.info(f"\nDataset saved to: {output_path}")
    logger.info(f"Final shape: {dataset.shape}")
    logger.info(f"\nLabel distribution:")
    print(dataset['label'].value_counts().to_string())
    logger.info(f"\nSource distribution:")
    print(dataset['source'].value_counts().to_string())
    logger.info("\nDone. Next: run src/feature_engineering.py")


if __name__ == "__main__":
    main()