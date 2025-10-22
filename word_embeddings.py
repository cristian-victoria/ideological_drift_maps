"""
Word Embedding Alignment for Semantic Drift Detection
Trains embeddings per decade and tracks term movement
Author: Cristian Victoria
"""

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("WORD EMBEDDING & SEMANTIC DRIFT ANALYSIS")
print("=" * 60)

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = 'preprocessed_manifestos.pkl'

# Key terms to track for semantic drift
KEY_TERMS = [
    'freedom', 'liberty', 'security', 'justice', 'equality',
    'government', 'economy', 'tax', 'welfare', 'education',
    'healthcare', 'environment', 'immigration', 'defense',
    'terrorism', 'trade', 'regulation', 'reform', 'rights'
]

# Decades to analyze
DECADES = [1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]

# Word2Vec parameters
VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2
EPOCHS = 10

print(f"\nTracking {len(KEY_TERMS)} key terms across {len(DECADES)} decades")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1/5] Loading preprocessed data...")
df = pd.read_pickle(INPUT_FILE)
print(f"Loaded {len(df)} manifestos")

# ============================================================================
# TRAIN EMBEDDINGS PER DECADE
# ============================================================================

print("\n[2/5] Training Word2Vec embeddings for each decade...")

decade_models = {}
decade_vocabularies = {}

for decade in DECADES:
    decade_df = df[df['decade'] == decade]
    
    if len(decade_df) == 0:
        print(f"  Skipping {decade}s - no data")
        continue
    
    # Get all tokens from this decade
    all_tokens = decade_df['lemmatized_tokens'].tolist()
    
    # Train Word2Vec model
    print(f"  Training {decade}s model...", end=" ")
    model = Word2Vec(
        sentences=all_tokens,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        epochs=EPOCHS,
        workers=4,
        seed=42
    )
    
    decade_models[decade] = model
    decade_vocabularies[decade] = set(model.wv.key_to_index.keys())
    
    print(f"(vocab size: {len(model.wv.key_to_index)})")

print(f"Trained {len(decade_models)} decade models")

# ============================================================================
# COMPUTE SEMANTIC DRIFT SCORES
# ============================================================================

print("\n[3/5] Computing semantic drift scores...")

drift_results = []

# Compare consecutive decades
decade_list = sorted(decade_models.keys())

for i in range(len(decade_list) - 1):
    decade1 = decade_list[i]
    decade2 = decade_list[i + 1]
    
    model1 = decade_models[decade1]
    model2 = decade_models[decade2]
    
    print(f"\n  Comparing {decade1}s → {decade2}s:")
    
    for term in KEY_TERMS:
        # Check if term exists in both models
        if term in model1.wv and term in model2.wv:
            vec1 = model1.wv[term]
            vec2 = model2.wv[term]
            
            # Compute cosine similarity
            similarity = 1 - cosine(vec1, vec2)
            drift_score = 1 - similarity  # Higher = more drift
            
            drift_results.append({
                'term': term,
                'decade1': decade1,
                'decade2': decade2,
                'similarity': similarity,
                'drift_score': drift_score
            })
            
            if drift_score > 0.3:  # Highlight significant drifts
                print(f"    {term:15s}: drift = {drift_score:.3f} ⚠️")
        else:
            print(f"    {term:15s}: not in both vocabularies ⊗")

drift_df = pd.DataFrame(drift_results)

# ============================================================================
# IDENTIFY TOP DRIFTING TERMS
# ============================================================================

print("\n[4/5] Identifying terms with largest semantic drift...")

# Average drift per term across all time periods
term_avg_drift = drift_df.groupby('term')['drift_score'].mean().sort_values(ascending=False)

print("\nTop 10 terms with highest average drift:")
print("-" * 60)
for i, (term, drift) in enumerate(term_avg_drift.head(10).items(), 1):
    print(f"{i:2d}. {term:15s}: {drift:.4f}")

# Save drift results
drift_df.to_csv('semantic_drift_scores.csv', index=False)
print("\nSaved semantic_drift_scores.csv")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n[5/5] Creating visualizations...")

# Visualization 1: Drift heatmap
print("  Creating drift heatmap...")

plt.figure(figsize=(14, 8))

# Create pivot table for heatmap
pivot_data = []
for term in KEY_TERMS:
    term_data = drift_df[drift_df['term'] == term]
    row = [term]
    for decade in decade_list[:-1]:
        matches = term_data[term_data['decade1'] == decade]
        if len(matches) > 0:
            row.append(matches['drift_score'].values[0])
        else:
            row.append(np.nan)
    pivot_data.append(row)

columns = ['term'] + [f"{d}s→{d+10}s" for d in decade_list[:-1]]
heatmap_df = pd.DataFrame(pivot_data, columns=columns)
heatmap_df = heatmap_df.set_index('term')

sns.heatmap(heatmap_df, cmap='YlOrRd', annot=True, fmt='.2f', 
            cbar_kws={'label': 'Drift Score'}, linewidths=0.5)
plt.title('Semantic Drift of Key Terms Across Decades', fontsize=16, pad=20)
plt.xlabel('Time Period Transition', fontsize=12)
plt.ylabel('Term', fontsize=12)
plt.tight_layout()
plt.savefig('drift_heatmap.png', dpi=300, bbox_inches='tight')
print("  Saved drift_heatmap.png")
plt.close()

# Visualization 2: Top drifting terms over time
print("  Creating drift timeline...")

top_terms = term_avg_drift.head(5).index.tolist()

plt.figure(figsize=(12, 6))
for term in top_terms:
    term_data = drift_df[drift_df['term'] == term]
    x = [f"{row['decade1']}-{row['decade2']}" for _, row in term_data.iterrows()]
    y = term_data['drift_score'].values
    plt.plot(x, y, marker='o', linewidth=2, label=term, markersize=8)

plt.xlabel('Decade Transition', fontsize=12)
plt.ylabel('Drift Score', fontsize=12)
plt.title('Semantic Drift Timeline for Top 5 Terms', fontsize=14, pad=15)
plt.legend(loc='best', fontsize=10)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('drift_timeline.png', dpi=300, bbox_inches='tight')
print("  Saved drift_timeline.png")
plt.close()

# Visualization 3: 2D projection of term evolution
print("  Creating term evolution map...")

# Focus on one interesting term (e.g., 'security')
focus_term = 'security'

if focus_term in model1.wv:
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(decade_models)))
    
    for i, (decade, model) in enumerate(sorted(decade_models.items())):
        if focus_term in model.wv:
            # Get vector for focus term
            vec = model.wv[focus_term].reshape(1, -1)
            
            # Simple 2D projection (just first 2 dimensions for visualization)
            x, y = vec[0][0], vec[0][1]
            
            plt.scatter(x, y, c=[colors[i]], s=200, alpha=0.7, 
                       edgecolors='black', linewidth=2)
            plt.text(x+0.01, y+0.01, f"{decade}s", fontsize=10)
    
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.title(f'Evolution of "{focus_term}" Across Decades', fontsize=14, pad=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('term_evolution_map.png', dpi=300, bbox_inches='tight')
    print("  Saved term_evolution_map.png")
    plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 60)
print("SEMANTIC DRIFT ANALYSIS COMPLETE!")
print("=" * 60)

print(f"\nTotal drift measurements: {len(drift_df)}")
print(f"Average drift score: {drift_df['drift_score'].mean():.4f}")
print(f"Max drift observed: {drift_df['drift_score'].max():.4f}")
print(f"  Term: {drift_df.loc[drift_df['drift_score'].idxmax(), 'term']}")
print(f"  Period: {drift_df.loc[drift_df['drift_score'].idxmax(), 'decade1']}s → {drift_df.loc[drift_df['drift_score'].idxmax(), 'decade2']}s")

print("\n" + "=" * 60)
print("FILES CREATED:")
print("=" * 60)
print("1. semantic_drift_scores.csv - All drift measurements")
print("2. drift_heatmap.png - Visual heatmap of term drift")
print("3. drift_timeline.png - Timeline of top drifting terms")
print("4. term_evolution_map.png - 2D projection of term evolution")
print("\n" + "=" * 60)