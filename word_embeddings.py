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

# Visualization 3: 2D projection of term evolution (IMPROVED - SHOWS MULTIPLE TERMS)
print("  Creating term evolution map...")

# Select top N terms with highest drift that appear in most decades
N_TERMS_TO_SHOW = 5

# Find terms that appear in at least 6 decades
term_decade_counts = {}
for term in KEY_TERMS:
    count = sum(1 for decade in decade_list if term in decade_models[decade].wv)
    term_decade_counts[term] = count

# Get top drifting terms that appear in enough decades
eligible_terms = [term for term in term_avg_drift.head(10).index 
                  if term_decade_counts.get(term, 0) >= 6][:N_TERMS_TO_SHOW]

if len(eligible_terms) > 0:
    print(f"  Visualizing evolution of: {', '.join(eligible_terms)}")
    
    # Collect all vectors for PCA
    all_vectors = []
    vector_labels = []
    
    for term in eligible_terms:
        for decade in decade_list:
            if term in decade_models[decade].wv:
                vec = decade_models[decade].wv[term]
                all_vectors.append(vec)
                vector_labels.append((term, decade))
    
    # Apply PCA to reduce to 2D
    all_vectors = np.array(all_vectors)
    pca = PCA(n_components=2, random_state=42)
    vectors_2d = pca.fit_transform(all_vectors)
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Use distinct colors for each term
    colors = plt.cm.tab10(np.linspace(0, 1, len(eligible_terms)))
    
    for term_idx, term in enumerate(eligible_terms):
        # Get all points for this term
        term_points = []
        term_decades = []
        
        for i, (label_term, label_decade) in enumerate(vector_labels):
            if label_term == term:
                term_points.append(vectors_2d[i])
                term_decades.append(label_decade)
        
        if len(term_points) > 0:
            term_points = np.array(term_points)
            
            # Plot trajectory line
            plt.plot(term_points[:, 0], term_points[:, 1], 
                    c=colors[term_idx], linewidth=2, alpha=0.6, linestyle='--')
            
            # Plot points for each decade
            for i, (point, decade) in enumerate(zip(term_points, term_decades)):
                marker_size = 150 + (i * 30)  # Larger for more recent decades
                plt.scatter(point[0], point[1], c=[colors[term_idx]], 
                           s=marker_size, alpha=0.7, edgecolors='black', 
                           linewidth=1.5, label=term if i == 0 else "")
                
                # Add decade label
                plt.annotate(f"{decade}s", (point[0], point[1]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.title(f'Semantic Evolution of Top {len(eligible_terms)} Drifting Terms Across Decades', 
             fontsize=14, pad=15)
    plt.legend(loc='best', fontsize=10, title='Terms')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('term_evolution_map.png', dpi=300, bbox_inches='tight')
    print("  Saved term_evolution_map.png")
    plt.close()
    
    # BONUS: Create individual evolution maps for each top term
    print("  Creating individual term evolution maps...")
    
    for term in eligible_terms[:3]:  # Top 3 terms
        plt.figure(figsize=(10, 8))
        
        term_points = []
        term_decades = []
        
        for i, (label_term, label_decade) in enumerate(vector_labels):
            if label_term == term:
                term_points.append(vectors_2d[i])
                term_decades.append(label_decade)
        
        if len(term_points) > 0:
            term_points = np.array(term_points)
            
            # Color gradient from old to new
            colors_gradient = plt.cm.viridis(np.linspace(0, 1, len(term_points)))
            
            # Plot trajectory
            plt.plot(term_points[:, 0], term_points[:, 1], 
                    c='gray', linewidth=2, alpha=0.5, linestyle='--', zorder=1)
            
            # Plot points
            for i, (point, decade) in enumerate(zip(term_points, term_decades)):
                size = 200 + (i * 50)
                plt.scatter(point[0], point[1], c=[colors_gradient[i]], 
                           s=size, alpha=0.8, edgecolors='black', 
                           linewidth=2, zorder=2)
                plt.annotate(f"{decade}s", (point[0], point[1]), 
                           fontsize=11, fontweight='bold',
                           xytext=(8, 8), textcoords='offset points')
            
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
            plt.title(f'Evolution of "{term}" Across Decades', fontsize=14, pad=15)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'term_evolution_{term}.png', dpi=300, bbox_inches='tight')
            print(f"    Saved term_evolution_{term}.png")
            plt.close()

else:
    print("  Warning: Not enough terms with sufficient decade coverage for visualization")

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
print("4. term_evolution_map.png - 2D projection comparing multiple terms")
print("5. term_evolution_<term>.png - Individual evolution maps for top 3 terms")
print("\n" + "=" * 60)