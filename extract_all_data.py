"""
Combined extraction: PDFs (already done) + CSVs
Author: Cristian Victoria
"""
import os
import pandas as pd
import re
import glob

print("=" * 60)
print("COMBINING PDF + CSV DATA")
print("=" * 60)

manifestos_data = []

# Load existing PDF extractions
print("\n[1/3] Loading existing PDF data...")
if os.path.exists('extracted_manifestos.pkl'):
    pdf_df = pd.read_pickle('extracted_manifestos.pkl')
    print(f"Loaded {len(pdf_df)} PDF manifestos")
    
    # Add source column if it doesn't exist
    if 'source' not in pdf_df.columns:
        pdf_df['source'] = 'PDF'
    
    manifestos_data.extend(pdf_df.to_dict('records'))
else:
    print("No PDF data found - run extract_text.py first")

# Process Democratic CSVs
print("\n[2/3] Processing Democratic CSV files...")
dem_csvs = sorted(glob.glob('manifestos/democratic/*.csv'))
print(f"Found {len(dem_csvs)} CSV files")

for i, filepath in enumerate(dem_csvs, 1):
    filename = os.path.basename(filepath)
    print(f"  [{i}/{len(dem_csvs)}] {filename}...", end=" ")
    
    try:
        df = pd.read_csv(filepath)
        text = ' '.join(df['text'].dropna().astype(str))
        
        year_match = re.search(r'(\d{4})', filename)
        year = int(year_match.group(1)) if year_match else None
        
        manifestos_data.append({
            'party': 'Democratic Party',
            'year': year,
            'decade': (year // 10) * 10 if year else None,
            'filename': filename,
            'source': 'CSV',
            'text': text,
            'char_count': len(text),
            'word_count': len(text.split())
        })
        
        print(f"({len(text.split())} words)")
        
    except Exception as e:
        print(f"ERROR: {e}")

# Process Republican CSVs
print("\n[3/3] Processing Republican CSV files...")
rep_csvs = sorted(glob.glob('manifestos/republican/*.csv'))
print(f"Found {len(rep_csvs)} CSV files")

for i, filepath in enumerate(rep_csvs, 1):
    filename = os.path.basename(filepath)
    print(f"  [{i}/{len(rep_csvs)}] {filename}...", end=" ")
    
    try:
        df = pd.read_csv(filepath)
        text = ' '.join(df['text'].dropna().astype(str))
        
        year_match = re.search(r'(\d{4})', filename)
        year = int(year_match.group(1)) if year_match else None
        
        manifestos_data.append({
            'party': 'Republican Party',
            'year': year,
            'decade': (year // 10) * 10 if year else None,
            'filename': filename,
            'source': 'CSV',
            'text': text,
            'char_count': len(text),
            'word_count': len(text.split())
        })
        
        print(f"({len(text.split())} words)")
        
    except Exception as e:
        print(f"ERROR: {e}")

# Create final dataset
print("\n" + "=" * 60)
print("Creating final combined dataset...")
df = pd.DataFrame(manifestos_data)
df = df.sort_values('year')

# Filter out empty ones
df = df[df['word_count'] > 100]

print("\n" + "=" * 60)
print("FINAL DATASET READY!")
print("=" * 60)
print(f"Total manifestos: {len(df)}")
print(f"\nBy party:")
print(df['party'].value_counts())
print(f"\nBy source:")
if 'source' in df.columns:
    print(df['source'].value_counts())
else:
    print("  (source column not available)")
print(f"\nBy decade:")
print(df['decade'].value_counts().sort_index())
print(f"\nYear range: {df['year'].min()} - {df['year'].max()}")
print(f"\nTotal words: {df['word_count'].sum():,}")
print(f"Average words: {df['word_count'].mean():.0f}")

# Save final dataset
OUTPUT = 'final_manifestos_dataset.csv'
df.to_csv(OUTPUT, index=False)
df.to_pickle(OUTPUT.replace('.csv', '.pkl'))
print(f"\nSaved as {OUTPUT}")
print("=" * 60)
print("\nYOU'RE READY TO START ANALYSIS!")
print(f"Use '{OUTPUT}' for your temporal text mining")