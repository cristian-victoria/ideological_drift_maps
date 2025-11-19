# Makefile for Temporal Text Mining Pipeline
# Author: Cristian Victoria
# Usage: 
#	make        - Run full analysis pipeline
#   make clean  - Delete all generated files
#	make clean-visualizations - Delete only visualization files, keep extracted data

# Python interpreter
PYTHON = python3

# Main analysis script
MAIN_SCRIPT = run_analysis.py

# Data files to clean
DATA_FILES = decade_*.csv \
			extracted_manifestos.csv \
			extracted_manifestos.pkl \
			final_manifestos_dataset.csv \
			final_manifestos_dataset.pkl \
			preprocessed_manifestos.csv \
			preprocessed_manifestos.pkl
             

# Analysis result files
RESULT_FILES = discovered_topics.csv \
			semantic_drift_scores.csv \
			topic_evolution_by_decade.csv

# Visualization files
VIZ_FILES = drift_heatmap.png \
            drift_timeline.png \
            term_evolution_*.png \
            topic_heatmap.png \
            topic_timeline.png \
            party_topic_comparison.png \

# Default target - runs the full pipeline
.PHONY: all
all:
	@echo "Running Temporal Text Mining Pipeline..."
	@$(PYTHON) $(MAIN_SCRIPT)

# Run the analysis (same as 'make all')
.PHONY: run
run: all

# Clean all generated files
.PHONY: clean
clean:
	@echo "Cleaning generated files..."
	@echo "Removing data files..."
	@rm -f $(DATA_FILES)
	@echo "Removing analysis results..."
	@rm -f $(RESULT_FILES)
	@echo "Removing visualizations..."
	@rm -f $(VIZ_FILES)
	@echo "Cleanup complete!"

# Clean only output files, keep extracted data
.PHONY: clean-visualizations
clean-visualizations:
	@echo "Cleaning visualization files only (keeping extracted data)..."
	@rm -f $(VIZ_FILES)
	@echo "Output files cleaned!"