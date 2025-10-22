"""
Text Preprocessing Script
Cleans and prepares manifesto text for temporal analysis
Author: Cristian Victoria
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

print("=" * 60)
print("TEXT PREPROCESSING SCRIPT")
print("=" * 60)

# Download required NLTK data
print("\n[0/6] Downloading NLTK resources...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    print("NLTK resources ready")
except:
    print("Some NLTK downloads may have failed, but continuing...")

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = 'final_manifestos_dataset.csv'
OUTPUT_FILE = 'preprocessed_manifestos.csv'

# Words to track for semantic drift analysis
KEY_IDEOLOGICAL_TERMS = [
    'freedom', 'liberty', 'security', 'justice', 'equality',
    'democracy', 'government', 'economy', 'tax', 'welfare',
    'education', 'healthcare', 'environment', 'immigration',
    'defense', 'terrorism', 'trade', 'regulation', 'reform',
    'rights', 'conservative', 'liberal', 'family', 'jobs'
]

print(f"\nTracking {len(KEY_IDEOLOGICAL_TERMS)} key ideological terms")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1/6] Loading data...")
try:
    df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    print(f"Loaded {len(df)} manifestos")
except Exception as e:
    print(f"ERROR: {e}")
    exit()

# ============================================================================
# TEXT CLEANING FUNCTIONS
# ============================================================================

def clean_text(text):
    """Clean raw text"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove numbers (but keep words with numbers like "401k")
    text = re.sub(r'\b\d+\b', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def tokenize_and_filter(text, remove_stopwords=True):
    """Tokenize and filter text"""
    if not text:
        return []
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove punctuation and non-alphabetic tokens
    tokens = [token for token in tokens if token.isalpha()]
    
    # Remove stopwords (optional)
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        # Keep some important political words that are in stopwords
        keep_words = {'not', 'no', 'against', 'will', 'can', 'should', 'must'}
        stop_words = stop_words - keep_words
        tokens = [token for token in tokens if token not in stop_words]
    
    # Remove very short tokens
    tokens = [token for token in tokens if len(token) > 2]
    
    return tokens

def lemmatize_tokens(tokens):
    """Lemmatize tokens to their base form"""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

# ============================================================================
# PREPROCESSING
# ============================================================================

print("\n[2/6] Cleaning text...")
df['cleaned_text'] = df['text'].apply(clean_text)
print("Text cleaned")

print("\n[3/6] Tokenizing...")
df['tokens'] = df['cleaned_text'].apply(tokenize_and_filter)
print("Text tokenized")

print("\n[4/6] Lemmatizing...")
df['lemmatized_tokens'] = df['tokens'].apply(lemmatize_tokens)
print("Tokens lemmatized")

# Create processed text (joined tokens for easy use)
df['processed_text'] = df['lemmatized_tokens'].apply(lambda x: ' '.join(x))

# Calculate statistics
df['token_count'] = df['lemmatized_tokens'].apply(len)
df['unique_tokens'] = df['lemmatized_tokens'].apply(lambda x: len(set(x)))

# ============================================================================
# KEY TERM EXTRACTION
# ============================================================================

print("\n[5/6] Extracting key ideological terms...")

# Count occurrences of key terms in each manifesto
for term in KEY_IDEOLOGICAL_TERMS:
    df[f'count_{term}'] = df['lemmatized_tokens'].apply(
        lambda tokens: tokens.count(term)
    )

print(f"Tracked {len(KEY_IDEOLOGICAL_TERMS)} terms across all manifestos")

# ============================================================================
# CREATE DECADE-BASED SUBSETS
# ============================================================================

print("\n[6/6] Creating decade-based splits...")

# Group by decade
decade_groups = df.groupby('decade')

decade_summary = []
for decade, group in decade_groups:
    total_tokens = group['token_count'].sum()
    avg_tokens = group['token_count'].mean()
    
    decade_summary.append({
        'decade': decade,
        'num_manifestos': len(group),
        'total_tokens': total_tokens,
        'avg_tokens_per_manifesto': int(avg_tokens)
    })
    
    # Save decade-specific data
    decade_file = f'decade_{int(decade)}.csv'
    group.to_csv(decade_file, index=False)
    print(f"  Saved {decade_file} ({len(group)} manifestos, {total_tokens:,} tokens)")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "=" * 60)
print("SAVING PREPROCESSED DATA...")
print("=" * 60)

# Save full preprocessed dataset
df.to_csv(OUTPUT_FILE, index=False)
df.to_pickle(OUTPUT_FILE.replace('.csv', '.pkl'))
print(f"Saved {OUTPUT_FILE}")

# Save decade summary
decade_df = pd.DataFrame(decade_summary)
decade_df.to_csv('decade_summary.csv', index=False)
print("Saved decade_summary.csv")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE!")
print("=" * 60)

print(f"\nTotal manifestos processed: {len(df)}")
print(f"Total tokens: {df['token_count'].sum():,}")
print(f"Average tokens per manifesto: {df['token_count'].mean():.0f}")
print(f"Average unique tokens: {df['unique_tokens'].mean():.0f}")

print("\n" + "-" * 60)
print("TOKENS BY PARTY:")
print("-" * 60)
party_stats = df.groupby('party').agg({
    'token_count': ['sum', 'mean'],
    'unique_tokens': 'mean'
})
print(party_stats)

print("\n" + "-" * 60)
print("TOKENS BY DECADE:")
print("-" * 60)
print(decade_df.to_string(index=False))

print("\n" + "-" * 60)
print("TOP 10 MOST FREQUENT KEY TERMS (across all manifestos):")
print("-" * 60)
term_counts = []
for term in KEY_IDEOLOGICAL_TERMS:
    total = df[f'count_{term}'].sum()
    term_counts.append((term, total))

term_counts.sort(key=lambda x: x[1], reverse=True)
for i, (term, count) in enumerate(term_counts[:10], 1):
    print(f"{i:2d}. {term:15s} : {count:5d} occurrences")

print("\n" + "=" * 60)
print("FILES CREATED:")
print("=" * 60)
print(f"1. {OUTPUT_FILE} - Full preprocessed dataset")
print(f"2. {OUTPUT_FILE.replace('.csv', '.pkl')} - Pickle format (faster loading)")
print("3. decade_summary.csv - Summary statistics by decade")
print("4. decade_XXXX.csv - Individual files for each decade")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE - READY FOR ANALYSIS!")
print("=" * 60)