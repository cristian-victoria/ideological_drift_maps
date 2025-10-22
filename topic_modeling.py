"""
Dynamic Topic Modeling Analysis
Tracks evolution of policy topics across decades
Author: Cristian Victoria
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TOPIC MODELING ANALYSIS")
print("=" * 60)

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = 'preprocessed_manifestos.pkl'
N_TOPICS = 4  # Number of topics to extract (optimized for 40 documents)
N_TOP_WORDS = 10  # Top words per topic
DECADES = [1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]

print(f"\nExtracting {N_TOPICS} topics with top {N_TOP_WORDS} words each")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1/6] Loading preprocessed data...")
df = pd.read_pickle(INPUT_FILE)
print(f"Loaded {len(df)} manifestos")

# ============================================================================
# PREPARE DATA FOR LDA
# ============================================================================

print("\n[2/6] Preparing documents for topic modeling...")

documents = df['processed_text'].tolist()

# Presidential and candidate names to filter (already in preprocessing, but double-check)
NAMES_TO_REMOVE = [
    'clinton', 'obama', 'trump', 'biden', 'bush', 'reagan', 'ronald',
    'carter', 'nixon', 'eisenhower', 'kennedy', 'johnson', 'ford',
    'gore', 'romney', 'mccain', 'kerry', 'dole', 'dukakis', 'mondale',
    'humphrey', 'goldwater', 'roosevelt', 'truman', 'george', 'donald',
    'hillary', 'bill', 'barack', 'mitt', 'john', 'al', 'joe', 'kamala',
    'harris', 'pence', 'cheney', 'palin', 'ryan', 'kaine'
]

# Generic/filler words that don't contribute to topic meaning
GENERIC_STOPWORDS = [
    'shall', 'will', 'must', 'can', 'may', 'like', 'make', 'get',
    'current', 'new', 'year', 'time', 'also', 'well', 'way', 'need',
    'tion', 'ment', 'ing',  # OCR artifacts
    'today', 'tomorrow', 'future', 'past', 'now',  # temporal words
    'say', 'said', 'call', 'called', 'ask', 'tell',  # communication verbs
    'thing', 'things', 'much', 'many', 'more', 'most', 'every', 'all'  # quantifiers
]

# Create vectorizer with optimized parameters
vectorizer = CountVectorizer(
    max_features=2500,  # Keep top 2500 words (sufficient for 40 docs)
    min_df=3,  # Word must appear in at least 3 documents (filters rare words)
    max_df=0.7,  # Ignore words in more than 70% of documents (filters common words)
    stop_words=NAMES_TO_REMOVE + GENERIC_STOPWORDS  # Combined stopwords list
)

doc_term_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

print(f"Created document-term matrix: {doc_term_matrix.shape}")
print(f"  Vocabulary size: {len(feature_names)}")

# ============================================================================
# TRAIN LDA MODEL (OVERALL)
# ============================================================================

print("\n[3/6] Training overall LDA model...")

lda_model = LatentDirichletAllocation(
    n_components=N_TOPICS,
    max_iter=50,  # Sufficient iterations for convergence
    learning_method='online',  # Faster for small datasets
    random_state=42,  # Reproducibility
    n_jobs=-1  # Use all CPU cores
)

lda_model.fit(doc_term_matrix)
print("LDA model trained")

# Get topic-word distributions
topic_word_dist = lda_model.components_

# ============================================================================
# EXTRACT AND DISPLAY TOPICS
# ============================================================================

print("\n[4/6] Extracting topics...")

def get_top_words(topic_idx, n_words=10):
    """Get top n words for a given topic"""
    top_word_indices = topic_word_dist[topic_idx].argsort()[-n_words:][::-1]
    return [feature_names[i] for i in top_word_indices]

def label_topic(words):
    """
    Improved heuristic to label topics based on keyword matching.
    Uses expanded categories and comprehensive keyword lists.
    """
    words_str = ' '.join(words).lower()
    
    # Define categories with comprehensive keyword lists
    scores = {
        'National Security': sum(1 for w in [
            'security', 'defense', 'military', 'terrorism', 'war', 'terrorist', 
            'armed', 'terror', 'iraq', 'afghanistan', 'homeland', 'threat',
            'soldier', 'veteran', 'intelligence', 'weapon', 'nuclear'
        ] if w in words_str),
        
        'Economy & Jobs': sum(1 for w in [
            'economy', 'job', 'tax', 'business', 'trade', 'economic', 
            'employment', 'taxpayer', 'wage', 'worker', 'income', 'growth',
            'market', 'industry', 'corporation', 'fiscal', 'budget'
        ] if w in words_str),
        
        'Education': sum(1 for w in [
            'education', 'school', 'student', 'teacher', 'college', 'university',
            'learning', 'classroom', 'academic', 'scholarship', 'diploma'
        ] if w in words_str),
        
        'Healthcare': sum(1 for w in [
            'healthcare', 'health', 'medical', 'insurance', 'patient', 'hospital',
            'medicaid', 'medicare', 'abortion', 'doctor', 'care', 'coverage',
            'prescription', 'disease', 'clinic'
        ] if w in words_str),
        
        'Environment & Energy': sum(1 for w in [
            'environment', 'energy', 'climate', 'clean', 'environmental', 
            'renewable', 'pollution', 'gas', 'oil', 'coal', 'solar', 'wind',
            'emission', 'carbon', 'conservation', 'sustainable'
        ] if w in words_str),
        
        'Social Justice & Civil Rights': sum(1 for w in [
            'black', 'tribal', 'native', 'color', 'racial', 'discrimination',
            'minority', 'equality', 'civil', 'rights', 'justice', 'african',
            'latino', 'asian', 'indigenous', 'lgbtq', 'women', 'gender'
        ] if w in words_str),
        
        'Infrastructure & Investment': sum(1 for w in [
            'infrastructure', 'invest', 'funding', 'affordable', 'housing',
            'transportation', 'build', 'construction', 'bridge', 'road',
            'broadband', 'internet', 'water', 'public'
        ] if w in words_str),
        
        'Social Policy & Welfare': sum(1 for w in [
            'social', 'welfare', 'family', 'child', 'parent', 'community',
            'poverty', 'assistance', 'benefit', 'elderly', 'senior', 'disability'
        ] if w in words_str),
        
        'Immigration': sum(1 for w in [
            'immigration', 'immigrant', 'border', 'refugee', 'asylum',
            'citizenship', 'undocumented', 'deportation', 'visa'
        ] if w in words_str),
        
        'Government & Reform': sum(1 for w in [
            'government', 'law', 'reform', 'regulation', 'policy', 'federal',
            'congress', 'senate', 'legislative', 'constitutional', 'democracy',
            'vote', 'election', 'transparency', 'accountability'
        ] if w in words_str),
    }
    
    # Get category with highest score
    max_score = max(scores.values())
    
    # Require at least 2 keyword matches for confident labeling
    if max_score >= 2:
        return max(scores.items(), key=lambda x: x[1])[0]
    else:
        # If no clear match, examine specific patterns
        if any(w in words_str for w in ['funding', 'affordable', 'invest']):
            return 'Infrastructure & Investment'
        elif any(w in words_str for w in ['assure', 'ensure', 'support']):
            return 'General Policy'
        else:
            return 'Mixed Policy'

print("\nDiscovered Topics:")
print("-" * 60)

topic_labels = []
topic_details = []

for topic_idx in range(N_TOPICS):
    top_words = get_top_words(topic_idx, N_TOP_WORDS)
    label = label_topic(top_words)
    topic_labels.append(label)
    
    print(f"\nTopic {topic_idx + 1}: {label}")
    print(f"  Top words: {', '.join(top_words)}")
    
    topic_details.append({
        'topic_id': topic_idx + 1,
        'label': label,
        'top_words': ', '.join(top_words)
    })

# Save topic descriptions
topics_df = pd.DataFrame(topic_details)
topics_df.to_csv('discovered_topics.csv', index=False)
print("\nSaved discovered_topics.csv")

# ============================================================================
# COMPUTE TOPIC DISTRIBUTIONS PER DECADE
# ============================================================================

print("\n[5/6] Computing topic distributions per decade...")

# Get document-topic distributions
doc_topic_dist = lda_model.transform(doc_term_matrix)

# Add topic distributions to dataframe
for topic_idx in range(N_TOPICS):
    df[f'topic_{topic_idx + 1}'] = doc_topic_dist[:, topic_idx]

# Compute average topic prominence per decade
decade_topic_data = []

for decade in DECADES:
    decade_df = df[df['decade'] == decade]
    
    if len(decade_df) == 0:
        continue
    
    row = {'decade': decade}
    for topic_idx in range(N_TOPICS):
        avg_prominence = decade_df[f'topic_{topic_idx + 1}'].mean()
        row[topic_labels[topic_idx]] = avg_prominence
    
    decade_topic_data.append(row)

decade_topics_df = pd.DataFrame(decade_topic_data)
decade_topics_df.to_csv('topic_evolution_by_decade.csv', index=False)
print("Saved topic_evolution_by_decade.csv")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n[6/6] Creating visualizations...")

# Visualization 1: Topic distribution heatmap
print("  Creating topic heatmap...")

plt.figure(figsize=(12, 8))

# Prepare data for heatmap
heatmap_data = decade_topics_df.set_index('decade').T

sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.2f',
            cbar_kws={'label': 'Topic Prominence'}, linewidths=0.5)
plt.title('Topic Prominence Across Decades', fontsize=16, pad=20)
plt.xlabel('Decade', fontsize=12)
plt.ylabel('Topic', fontsize=12)
plt.tight_layout()
plt.savefig('topic_heatmap.png', dpi=300, bbox_inches='tight')
print("  Saved topic_heatmap.png")
plt.close()

# Visualization 2: Topic evolution over time (line plot)
print("  Creating topic evolution timeline...")

plt.figure(figsize=(14, 8))

decades_list = decade_topics_df['decade'].values

# Use distinct colors for better visualization
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for i, topic_label in enumerate(set(topic_labels)):  # Use set to avoid duplicates
    if topic_label in decade_topics_df.columns:
        values = decade_topics_df[topic_label].values
        plt.plot(decades_list, values, marker='o', linewidth=2.5, 
                label=topic_label, markersize=8, color=colors[i % len(colors)])

plt.xlabel('Decade', fontsize=12)
plt.ylabel('Average Topic Prominence', fontsize=12)
plt.title('Evolution of Policy Topics Over Time', fontsize=16, pad=20)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(decades_list, [f"{d}s" for d in decades_list])
plt.tight_layout()
plt.savefig('topic_timeline.png', dpi=300, bbox_inches='tight')
print("  Saved topic_timeline.png")
plt.close()

# Visualization 3: Party comparison for key topics
print("  Creating party comparison...")

# Select top 4 most prominent topics overall
unique_labels = list(set(topic_labels))  # Get unique labels
topic_means = {label: decade_topics_df[label].mean() 
               for label in unique_labels if label in decade_topics_df.columns}
top_topics = sorted(topic_means.items(), key=lambda x: x[1], reverse=True)[:4]
top_labels = [t[0] for t in top_topics]

# Adjust subplot layout based on number of topics
n_topics = len(top_labels)
if n_topics == 4:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
elif n_topics == 3:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
elif n_topics == 2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
else:
    fig, axes = plt.subplots(1, n_topics, figsize=(6*n_topics, 5))
    if n_topics == 1:
        axes = [axes]

for idx, topic_label in enumerate(top_labels):
    if idx >= len(axes):
        break
    ax = axes[idx]
    
    # Get data by party
    for party in df['party'].unique():
        party_df = df[df['party'] == party]
        party_decade_data = []
        
        for decade in DECADES:
            decade_party_df = party_df[party_df['decade'] == decade]
            if len(decade_party_df) > 0:
                # Find which topic index has this label
                topic_idx = topic_labels.index(topic_label)
                topic_col = f'topic_{topic_idx + 1}'
                avg_val = decade_party_df[topic_col].mean()
                party_decade_data.append((decade, avg_val))
        
        if party_decade_data:
            decades_p, values_p = zip(*party_decade_data)
            party_name = 'Democrats' if 'Democratic' in party else 'Republicans'
            ax.plot(decades_p, values_p, marker='o', linewidth=2.5, 
                   label=party_name, markersize=8)
    
    ax.set_title(topic_label, fontsize=12, fontweight='bold')
    ax.set_xlabel('Decade', fontsize=10)
    ax.set_ylabel('Topic Prominence', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(DECADES)
    ax.set_xticklabels([f"{d}s" for d in DECADES], rotation=45, ha='right')

plt.suptitle('Party Comparison: Top Topics', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('party_topic_comparison.png', dpi=300, bbox_inches='tight')
print("  Saved party_topic_comparison.png")
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 60)
print("TOPIC MODELING COMPLETE!")
print("=" * 60)

print(f"\nNumber of topics: {N_TOPICS}")
print(f"Documents analyzed: {len(df)}")
print(f"Vocabulary size: {len(feature_names)}")

print("\nMost prominent topics (overall average):")
print("-" * 60)
topic_avg_prominence = {}
unique_topic_labels = list(set(topic_labels))  # Get unique labels only
for topic_label in unique_topic_labels:
    if topic_label in decade_topics_df.columns:
        avg = decade_topics_df[topic_label].mean()
        topic_avg_prominence[topic_label] = avg

for i, (topic, avg) in enumerate(sorted(topic_avg_prominence.items(), 
                                       key=lambda x: x[1], reverse=True), 1):
    print(f"{i}. {topic:30s}: {avg:.4f}")

print("\n" + "=" * 60)
print("FILES CREATED:")
print("=" * 60)
print("1. discovered_topics.csv - Topic descriptions")
print("2. topic_evolution_by_decade.csv - Topic prominence by decade")
print("3. topic_heatmap.png - Heatmap of topics across time")
print("4. topic_timeline.png - Line plot of topic evolution")
print("5. party_topic_comparison.png - Party differences on top topics")

print("\n" + "=" * 60)
print("TOPIC MODELING COMPLETE!")
print("=" * 60)