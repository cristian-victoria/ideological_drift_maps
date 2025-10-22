"""
PDF Text Extraction Script with OCR Support
Extracts text from manifesto PDFs (including scanned images)
Author: Cristian Victoria
"""

import os
import pandas as pd
import re
from pdf2image import convert_from_path
import pytesseract

print("=" * 60)
print("PDF TEXT EXTRACTION SCRIPT (WITH OCR)")
print("=" * 60)

# Configuration
MANIFESTO_DIR = 'manifestos'
OUTPUT_FILE = 'extracted_manifestos.csv'

manifestos_data = []

def extract_text_from_pdf_ocr(pdf_path):
    """Extract text from PDF using OCR"""
    try:
        print("      Converting to images...", end=" ")
        images = convert_from_path(pdf_path, dpi=300)
        print(f"✓ ({len(images)} pages)")
        
        text = ""
        for i, image in enumerate(images, 1):
            print(f"      OCR page {i}/{len(images)}...", end=" ")
            page_text = pytesseract.image_to_string(image)
            text += page_text + "\n"
            print("✓")
        
        return text.strip()
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return ""

# ============================================================================
# PROCESS DEMOCRATIC PARTY
# ============================================================================
print("\n[1/2] Processing Democratic Party manifestos...")
dem_dir = os.path.join(MANIFESTO_DIR, 'democratic')

if os.path.exists(dem_dir):
    dem_pdfs = [f for f in os.listdir(dem_dir) if f.endswith('.pdf')]
    print(f"  Found {len(dem_pdfs)} PDFs")
    
    for i, filename in enumerate(sorted(dem_pdfs), 1):
        filepath = os.path.join(dem_dir, filename)
        print(f"    [{i}/{len(dem_pdfs)}] {filename}")
        
        year_match = re.search(r'(\d{4})', filename)
        year = int(year_match.group(1)) if year_match else None
        
        text = extract_text_from_pdf_ocr(filepath)
        
        manifestos_data.append({
            'party': 'Democratic Party',
            'year': year,
            'decade': (year // 10) * 10 if year else None,
            'filename': filename,
            'source': 'PDF',
            'text': text,
            'char_count': len(text),
            'word_count': len(text.split())
        })
        
        print(f"      ✓ Complete! ({len(text.split())} words)\n")

# ============================================================================
# PROCESS REPUBLICAN PARTY
# ============================================================================
print("\n[2/2] Processing Republican Party manifestos...")
rep_dir = os.path.join(MANIFESTO_DIR, 'republican')

if os.path.exists(rep_dir):
    rep_pdfs = [f for f in os.listdir(rep_dir) if f.endswith('.pdf')]
    print(f"  Found {len(rep_pdfs)} PDFs")
    
    for i, filename in enumerate(sorted(rep_pdfs), 1):
        filepath = os.path.join(rep_dir, filename)
        print(f"    [{i}/{len(rep_pdfs)}] {filename}")
        
        year_match = re.search(r'(\d{4})', filename)
        year = int(year_match.group(1)) if year_match else None
        
        text = extract_text_from_pdf_ocr(filepath)
        
        manifestos_data.append({
            'party': 'Republican Party',
            'year': year,
            'decade': (year // 10) * 10 if year else None,
            'filename': filename,
            'source': 'PDF',
            'text': text,
            'char_count': len(text),
            'word_count': len(text.split())
        })
        
        print(f"      ✓ Complete! ({len(text.split())} words)\n")

# ============================================================================
# CREATE FINAL DATASET
# ============================================================================
print("\n[3/3] Creating dataset...")
df = pd.DataFrame(manifestos_data)
df = df.sort_values('year')

print("\n" + "=" * 60)
print("EXTRACTION COMPLETED!")
print("=" * 60)
print(f"Total manifestos: {len(df)}")
print(f"\nBy party:")
print(df['party'].value_counts())
print(f"\nBy decade:")
print(df['decade'].value_counts().sort_index())
print(f"\nYear range: {df['year'].min()} - {df['year'].max()}")
print(f"\nTotal words: {df['word_count'].sum():,}")

# Save
print(f"\nSaving to {OUTPUT_FILE}...")
df.to_csv(OUTPUT_FILE, index=False)
df.to_pickle(OUTPUT_FILE.replace('.csv', '.pkl'))
print("✓ Saved!")
print(f"✓ Also saved as {OUTPUT_FILE.replace('.csv', '.pkl')}")

print("\n" + "=" * 60)
print("✓ Ready!")
print("=" * 60)