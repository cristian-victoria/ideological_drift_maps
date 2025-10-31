# Tracing Ideological Drifts in Political Manifestos

**Author:** Cristian Victoria   
**Date:** 25 November 2025

## Project Overview

This project uses temporal text mining techniques to detect and quantify ideological drifts in US political manifestos from 1948-2024. By analyzing 40 manifestos from Democratic and Republican parties, we track how key ideological terms change meaning and how policy topics evolve over time.

## Dataset

- **Source:** Manifesto Project Corpus
- **Scope:** United States (1948-2024)
- **Parties:** Democratic Party (20 manifestos), Republican Party (20 manifestos)
- **Total Words:** 888,566 words
- **Coverage:** 9 decades (1940s-2020s)

## Installation

### Required Libraries
```bash
pip3 install pandas numpy gensim scikit-learn matplotlib seaborn scipy nltk PyPDF2 pdf2image pytesseract pillow
brew install tesseract poppler  # For Mac users
```

## Usage

### Quick Start (Run Everything)
```bash
make # or if you want to run it directly python3 run_analysis.py
```

### Step-by-Step Execution

If you prefer to run steps individually:
```bash
# Step 1: Extract and combine data
python3 extract_text.py # Extracts text from PDFs
python3 extract_all_data.py

# Step 2: Preprocess text
python3 preprocess_text.py

# Step 3: Word embedding analysis
python3 word_embeddings.py

# Step 4: Topic modeling
python3 topic_modeling.py

# Optional Final Step: Summary of Visualizations
python3 create_summary_visualizations.py
```

## Project Structure
```
manifestoData/
├── manifestos/                         # Raw PDF and CSV files
│   ├── democratic/
│   └── republican/
├── Makefile                            # Makefile to ease program interaction 
├── extract_text.py                     # Text extraction from PDF
├── extract_all_data.py                 # General data extraction
├── preprocess_text.py                  # Text preprocessing
├── word_embeddings.py                  # Semantic drift analysis
├── topic_modeling.py                   # Topic evolution analysis
├── run_analysis.py                     # Master pipeline script
├── README.md                           # This file
├── create_summary_visualiaztions.py    # Visualization summary
└── [output files]                      # Generated CSVs and PNGs
```

## Methodology

### 1. Text Preprocessing
- Tokenization and lemmatization
- Stopword removal (preserving political terms)
- Decade-based document splitting

### 2. Word Embedding Alignment
- Train Word2Vec models per decade (100 dimensions)
- Compute cosine similarity between decades
- Track 19 key ideological terms

### 3. Dynamic Topic Modeling
- Latent Dirichlet Allocation (8 topics)
- Track topic prominence over time
- Compare party differences

## Key Results

### Top Semantic Drift Terms

1. **Terrorism** (0.217) - Emerged post-9/11
2. **Tax** (0.216) - Major shifts during Reagan era
3. **Equality** (0.188) - Post-WWII civil rights evolution

### Discovered Topics

- Economy & Jobs
- National Security
- Environment & Energy
- Social Policy
- Healthcare
- Education
- Rights & Justice
- Government Reform

## Output Files

### Data Files
- `final_manifestos_dataset.csv` - Combined extracted text
- `preprocessed_manifestos.csv` - Cleaned and tokenized data
- `decade_*.csv` - Data split by decade

### Analysis Results
- `semantic_drift_scores.csv` - Drift measurements
- `discovered_topics.csv` - Topic descriptions
- `topic_evolution_by_decade.csv` - Topic trends

### Visualizations
- `drift_heatmap.png` - Semantic drift across decades
- `drift_timeline.png` - Top drifting terms over time
- `term_evolution_map.png` - 2D term evolution
- `topic_heatmap.png` - Topic prominence heatmap
- `topic_timeline.png` - Topic evolution timeline
- `party_topic_comparison.png` - Party differences

## References

- Blei, D. M., & Lafferty, J. D. (2006). Dynamic topic models.
- Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). Diachronic word embeddings reveal statistical laws of semantic change.
- Slapin, J. B., & Proksch, S. O. (2008). A scaling model for estimating time-series party positions from texts.

## License

Academic use only. Data from Manifesto Project Corpus.