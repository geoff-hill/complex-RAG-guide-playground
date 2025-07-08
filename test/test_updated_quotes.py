#!/usr/bin/env python
"""
Test the updated quote extraction function with smart quotes.
"""

import sys
from pathlib import Path

# Add current directory to path to import helper_functions
sys.path.append('.')

from helper_functions import split_into_chapters, extract_book_quotes_as_documents

def test_updated_quotes():
    """Test the updated quote extraction function."""
    print("🔍 Testing updated quote extraction function...")
    
    # Path to the PDF
    pdf_path = "Harry Potter - Book 1 - The Sorcerers Stone.pdf"
    
    if not Path(pdf_path).exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    try:
        # Split the PDF into chapters
        print("📖 Splitting PDF into chapters...")
        chapters = split_into_chapters(pdf_path)
        print(f"✅ Successfully split PDF into {len(chapters)} chapters")
        
        # Test quote extraction with different minimum lengths
        for min_length in [10, 20, 30, 50]:
            print(f"\n🔍 Extracting quotes with minimum length {min_length}:")
            quotes = extract_book_quotes_as_documents(chapters, min_length)
            print(f"  Found {len(quotes)} quotes")
            
            if quotes:
                print("  Sample quotes:")
                for i, quote_doc in enumerate(quotes[:5]):
                    print(f"    {i+1}. {quote_doc.page_content}")
                if len(quotes) > 5:
                    print(f"    ... and {len(quotes) - 5} more")
            else:
                print("  No quotes found")
        
        # Specifically look for the "Little tyke" quote
        print(f"\n🎯 Looking for 'Little tyke' quote:")
        quotes = extract_book_quotes_as_documents(chapters, 5)  # Lower threshold to catch short quotes
        
        found_little_tyke = False
        for quote_doc in quotes:
            if "Little tyke" in quote_doc.page_content:
                print(f"  ✅ Found: {quote_doc.page_content}")
                found_little_tyke = True
                break
        
        if not found_little_tyke:
            print("  ❌ 'Little tyke' quote not found")
        
        # Show some sample cleaned text from Chapter 1
        print(f"\n📖 Sample cleaned text from Chapter 1:")
        if chapters:
            chapter1_content = chapters[0].page_content
            sample = chapter1_content[:500]
            print(f"  {sample}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_updated_quotes() 