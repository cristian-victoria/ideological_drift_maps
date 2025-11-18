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

## Installation

### Required Libraries
```bash
pip3 install pandas numpy gensim scikit-learn matplotlib seaborn scipy nltk PyPDF2 pdf2image pytesseract pillow
brew install tesseract poppler  # For Mac users
```

## Usage

### Quick Start (Run Everything)
```bash
make # or if you want to run it directly: `python3 run_analysis.py`
```

### Step-by-Step Execution

If you prefer to run steps individually:
```bash
# Step 1: Extract text from scanned images
python3 extract_text.py 

# Step 2: Combine PDF and CSV data
python3 extract_all_data.py

# Step 3: Preprocess text
python3 preprocess_text.py

# Step 4: Word embedding analysis
python3 word_embeddings.py

# Step 5: Topic modeling
python3 topic_modeling.py
```

## Project Structure
```
manifestoData/
├── manifestos/                         # Raw PDF and CSV files
│   ├── democratic/
│   └── republican/
├── extract_all_data.py                 # Step 2: General data combination
├── extract_text.py                     # Step 1: Text extraction from PDF
├── Makefile                            # Makefile to ease interaction
├── preprocess_text.py                  # Step 3: Text preprocessing
├── README.md                           # THIS FILE
├── run_analysis.py                     # Master pipeline script
├── topic_modeling.py                   # Step 5: Topic evolution analysis
├── word_embeddings.py                  # Step 4: Semantic drift analysis
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

## Output Files

### Data Files
- `decade_*.csv` - Data split by decade
- `extracted_manifestos.csv` - Data from image pdf to readable text
- `extracted_manifestos.pkl` - Data from image pdf to readable text - used for combined data set
- `final_manifestos_dataset.csv` - Combined extracted text
- `final_manifestos_dataset.pkl` - Combined extracted text - used for processing
- `preprocessed_manifestos.csv` - Cleaned and tokenized data
- `preprocessed_manifestos.pkl` - Combined extracted text - used to train models

### Analysis Results
- `discovered_topics.csv` - Topic descriptions
- `semantic_drift_scores.csv` - Drift measurements
- `topic_evolution_by_decade.csv` - Topic trends

### Visualizations
- `drift_heatmap.png` - Semantic drift across decades
- `drift_timeline.png` - Top drifting terms over time
- `party_topic_comparison.png` - Party differences
- `term_evolution_*.png` - 2D term evolution
- `topic_heatmap.png` - Topic prominence heatmap
- `topic_timeline.png` - Topic evolution timeline

## References

- Blei, D. M., & Lafferty, J. D. (2006). Dynamic topic models.
- Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). Diachronic word embeddings reveal statistical laws of semantic change.
- Slapin, J. B., & Proksch, S. O. (2008). A scaling model for estimating time-series party positions from texts.

## License

Academic use only. Data from Manifesto Project Corpus.