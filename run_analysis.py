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
        'name': 'Data Extraction (PDFs + CSVs)',
        'script': 'extract_all_data.py',
        'duration_est': '1-2 minutes'
    },
    {
        'name': 'Text Preprocessing',
        'script': 'preprocess_text.py',
        'duration_est': '30-60 seconds'
    },
    {
        'name': 'Word Embedding Analysis',
        'script': 'word_embeddings.py',
        'duration_est': '2-3 minutes'
    },
    {
        'name': 'Topic Modeling',
        'script': 'topic_modeling.py',
        'duration_est': '1-2 minutes'
    }
]

print("\nThis pipeline will execute 4 major steps:")
for i, step in enumerate(STEPS, 1):
    print(f"  {i}. {step['name']} (~{step['duration_est']})")

print("\nEstimated total time: 5-8 minutes")
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
        print(f"\nERROR in Step {i}: {step['name']}")
        print(f"Script: {step['script']}")
        print("Please check the error messages above and fix before continuing.")
        exit(1)
    except FileNotFoundError:
        print(f"\nERROR: Script not found: {step['script']}")
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
    print(f"  {i}. {step['name']:40s}: {duration:.1f}s")

print("\n" + "=" * 70)
print("OUTPUT FILES CREATED:")
print("=" * 70)

# List all output files
output_files = [
    'Data Files:',
    '  - final_manifestos_dataset.csv',
    '  - preprocessed_manifestos.csv',
    '  - decade_*.csv (9 files)',
    '',
    'Analysis Results:',
    '  - semantic_drift_scores.csv',
    '  - discovered_topics.csv',
    '  - topic_evolution_by_decade.csv',
    '',
    'Visualizations:',
    '  - drift_heatmap.png',
    '  - drift_timeline.png',
    '  - term_evolution_map.png',
    '  - topic_heatmap.png',
    '  - topic_timeline.png',
    '  - party_topic_comparison.png'
]

for line in output_files:
    print(line)

print("\n" + "=" * 70)
print("All analysis complete! Ready for report writing.")
print("=" * 70)