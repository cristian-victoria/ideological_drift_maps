"""
Master Script - Runs Complete Analysis Pipeline
Executes all steps from data extraction to visualization
Author: Cristian Victoria
"""

import subprocess
import time
import os

print("=" * 70)
print(" " * 15 + "TEMPORAL TEXT MINING PIPELINE")
print(" " * 10 + "Ideological Drift in Political Manifestos")
print("=" * 70)

# Configuration
STEPS = [
    {
        'name': 'PDF Text Extraction (OCR)',
        'script': 'extract_text.py',
        'duration_est': '2-3 minutes',
        'optional': True  # Skip if already done
    },
    {
        'name': 'Data Combination (PDFs + CSVs)',
        'script': 'extract_all_data.py',
        'duration_est': '1-2 minutes',
        'optional': False
    },
    {
        'name': 'Text Preprocessing',
        'script': 'preprocess_text.py',
        'duration_est': '30-60 seconds',
        'optional': False
    },
    {
        'name': 'Word Embedding Analysis',
        'script': 'word_embeddings.py',
        'duration_est': '2-3 minutes',
        'optional': False
    },
    {
        'name': 'Topic Modeling',
        'script': 'topic_modeling.py',
        'duration_est': '1-2 minutes',
        'optional': False
    }
]

print("\nThis pipeline will execute 5 major steps:")
for i, step in enumerate(STEPS, 1):
    optional_tag = " (will skip if already done)" if step.get('optional') else ""
    print(f"  {i}. {step['name']} (~{step['duration_est']}){optional_tag}")

print("\nEstimated total time: 6-11 minutes")
print("=" * 70)

input("\nPress ENTER to start the analysis pipeline...")

# Track timing
start_time = time.time()
step_times = []

# Execute each step
for i, step in enumerate(STEPS, 1):
    print("\n" + "=" * 70)
    print(f"STEP {i}/{len(STEPS)}: {step['name']}")
    print("=" * 70)
    
    # Check if step can be skipped
    if step.get('optional'):
        if step['script'] == 'extract_text.py' and os.path.exists('extracted_manifestos.pkl'):
            print(f"\n✓ Skipping - extracted_manifestos.pkl already exists")
            print(f"  (Delete the file if you want to re-extract PDFs)")
            step_times.append(0)
            continue
    
    step_start = time.time()
    
    try:
        # Run the script
        result = subprocess.run(
            ['python3', step['script']],
            capture_output=False,
            text=True,
            check=True
        )
        
        step_end = time.time()
        step_duration = step_end - step_start
        step_times.append(step_duration)
        
        print(f"\n✓ Step {i} completed in {step_duration:.1f} seconds")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR in Step {i}: {step['name']}")
        print(f"Script: {step['script']}")
        print("Please check the error messages above and fix before continuing.")
        exit(1)
    except FileNotFoundError:
        print(f"\n✗ ERROR: Script not found: {step['script']}")
        print("Please ensure all scripts are in the current directory.")
        exit(1)

# Final summary
total_time = time.time() - start_time

print("\n" + "=" * 70)
print(" " * 20 + "PIPELINE COMPLETE!")
print("=" * 70)

print(f"\nTotal execution time: {total_time/60:.1f} minutes")

print("\nStep timings:")
for i, (step, duration) in enumerate(zip(STEPS, step_times), 1):
    if duration > 0:
        print(f"  {i}. {step['name']:40s}: {duration:.1f}s")
    else:
        print(f"  {i}. {step['name']:40s}: (skipped)")

print("\n" + "=" * 70)
print("OUTPUT FILES CREATED:")
print("=" * 70)

# List all output files organized by category
output_files = [
    '',
    'DATA FILES:',
    '  - extracted_manifestos.csv         (text from PDF manifestos)',
    '  - extracted_manifestos.pkl         (cached PDF data)',
    '  - final_manifestos_dataset.csv     (combined PDFs + CSVs, all 40 manifestos)',
    '  - preprocessed_manifestos.csv      (cleaned & tokenized text)',
    '  - preprocessed_manifestos.pkl      (cached preprocessed data)',
    '  - decade_1940.csv through decade_2020.csv  (9 decade-specific files)',
    '  - decade_summary.csv               (statistics by decade)',
    '',
    'ANALYSIS RESULTS:',
    '  - semantic_drift_scores.csv        (drift measurements for all terms)',
    '  - discovered_topics.csv            (LDA topic descriptions)',
    '  - topic_evolution_by_decade.csv    (topic prominence over time)',
    '',
    'VISUALIZATIONS - Semantic Drift:',
    '  - drift_heatmap.png                (all terms across all decades)',
    '  - drift_timeline.png               (top 5 drifting terms over time)',
    '  - term_evolution_map.png           (multi-term 2D PCA comparison)',
    '  - term_evolution_terrorism.png     (individual evolution map)',
    '  - term_evolution_tax.png           (individual evolution map)',
    '  - term_evolution_equality.png      (individual evolution map)',
    '     ↳ Note: Top 3 drifting terms get individual detailed maps',
    '',
    'VISUALIZATIONS - Topic Modeling:',
    '  - topic_heatmap.png                (topic prominence heatmap across decades)',
    '  - topic_timeline.png               (topic evolution timeline)',
    '  - party_topic_comparison.png       (Democrat vs Republican comparison)',
    ''
]

for line in output_files:
    print(line)

print("=" * 70)
print("✓ All analysis complete!")
print("=" * 70)