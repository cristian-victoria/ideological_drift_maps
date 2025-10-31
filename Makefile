# Makefile for Temporal Text Mining Pipeline
# Author: Cristian Victoria
# Usage: 
#   make        - Run full analysis pipeline
#   make clean  - Delete all generated files
#   make help   - Show available commands

# Python interpreter
PYTHON = python3

# Main analysis script
MAIN_SCRIPT = run_analysis.py

# Data files to clean
DATA_FILES = extracted_manifestos.csv \
             extracted_manifestos.pkl \
             final_manifestos_dataset.csv \
             final_manifestos_dataset.pkl \
             preprocessed_manifestos.csv \
             preprocessed_manifestos.pkl \
             decade_*.csv \
             decade_summary.csv

# Analysis result files
RESULT_FILES = semantic_drift_scores.csv \
               discovered_topics.csv \
               topic_evolution_by_decade.csv

# Visualization files
VIZ_FILES = drift_heatmap.png \
            drift_timeline.png \
            term_evolution_map.png \
            topic_heatmap.png \
            topic_timeline.png \
            party_topic_comparison.png \
            summary_visualization.png \
            summary_visualization_highres.png \
            key_findings_slide.png

# Default target - runs the full pipeline
.PHONY: all
all:
	@echo "=========================================="
	@echo "Running Temporal Text Mining Pipeline"
	@echo "=========================================="
	@$(PYTHON) $(MAIN_SCRIPT)

# Run the analysis (same as 'make all')
.PHONY: run
run: all

# Clean all generated files
.PHONY: clean
clean:
	@echo "=========================================="
	@echo "Cleaning generated files..."
	@echo "=========================================="
	@echo "Removing data files..."
	@rm -f $(DATA_FILES)
	@echo "Removing analysis results..."
	@rm -f $(RESULT_FILES)
	@echo "Removing visualizations..."
	@rm -f $(VIZ_FILES)
	@echo "✓ Cleanup complete!"

# Deep clean - removes everything including Python cache
.PHONY: deepclean
deepclean: clean
	@echo "Removing Python cache..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Deep cleanup complete!"

# Clean only output files, keep extracted data
.PHONY: clean-output
clean-output:
	@echo "Cleaning output files only (keeping extracted data)..."
	@rm -f preprocessed_manifestos.csv preprocessed_manifestos.pkl
	@rm -f decade_*.csv decade_summary.csv
	@rm -f $(RESULT_FILES)
	@rm -f $(VIZ_FILES)
	@echo "✓ Output files cleaned!"

# Show help
.PHONY: help
help:
	@echo "Temporal Text Mining Pipeline - Available Commands:"
	@echo ""
	@echo "  make              - Run the full analysis pipeline"
	@echo "  make run          - Same as 'make'"
	@echo "  make clean        - Delete all generated files"
	@echo "  make clean-output - Delete only results (keep extracted data)"
	@echo "  make deepclean    - Clean everything including Python cache"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Files cleaned by 'make clean':"
	@echo "  - All CSV and PKL data files"
	@echo "  - All analysis result files"
	@echo "  - All visualization PNG files"

# Check if required scripts exist
.PHONY: check
check:
	@echo "Checking required files..."
	@test -f extract_text.py && echo "✓ extract_text.py" || echo "✗ extract_text.py MISSING"
	@test -f extract_all_data.py && echo "✓ extract_all_data.py" || echo "✗ extract_all_data.py MISSING"
	@test -f preprocess_text.py && echo "✓ preprocess_text.py" || echo "✓ preprocess_text.py" || echo "✗ preprocess_text.py MISSING"
	@test -f word_embeddings.py && echo "✓ word_embeddings.py" || echo "✗ word_embeddings.py MISSING"
	@test -f topic_modeling.py && echo "✓ topic_modeling.py" || echo "✗ topic_modeling.py MISSING"
	@test -f run_analysis.py && echo "✓ run_analysis.py" || echo "✗ run_analysis.py MISSING"
	@test -d manifestos && echo "✓ manifestos/ directory" || echo "✗ manifestos/ directory MISSING"

# List all generated files
.PHONY: list
list:
	@echo "Generated files:"
	@echo ""
	@echo "Data files:"
	@ls -lh $(DATA_FILES) 2>/dev/null || echo "  (none found)"
	@echo ""
	@echo "Analysis results:"
	@ls -lh $(RESULT_FILES) 2>/dev/null || echo "  (none found)"
	@echo ""
	@echo "Visualizations:"
	@ls -lh $(VIZ_FILES) 2>/dev/null || echo "  (none found)"