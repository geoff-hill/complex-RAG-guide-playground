#!/usr/bin/env python
"""
Test for smart quotes and other quote characters in the text.
"""

import sys
import re
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from helper_functions import split_into_chapters

def test_smart_quotes():
    """Test for smart quotes and other quote characters."""
    print("🔍 Testing for smart quotes and other quote characters...")
    
    # Path to the PDF
    pdf_path = "Harry Potter - Book 1 - The Sorcerers Stone.pdf"
    
    if not Path(pdf_path).exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    try:
        # Split the PDF into chapters
        chapters = split_into_chapters(pdf_path)
        
        if len(chapters) == 0:
            print("❌ No chapters found")
            return
        
        # Get Chapter 1 content
        chapter1_content = chapters[0].page_content
        
        print(f"📊 Chapter 1 content length: {len(chapter1_content)} characters")
        
        # Check for different quote characters
        quote_chars = {
            'Straight double quotes': '"',
            'Straight single quotes': "'",
            'Smart quotes left': '"',
            'Smart quotes right': '"',
            'Smart single quotes left': ''',
            'Smart single quotes right': ''',
            'Em dash': '—',
            'En dash': '–',
            'Regular dash': '-'
        }
        
        print(f"\n🔍 Quote character analysis:")
        for char_name, char in quote_chars.items():
            count = chapter1_content.count(char)
            print(f"  {char_name} ('{char}'): {count} occurrences")
        
        # Look for the "Little tyke" area specifically
        print(f"\n🎯 Analyzing 'Little tyke' area:")
        
        if "Little" in chapter1_content and "tyke" in chapter1_content:
            little_pos = chapter1_content.find("Little")
            tyke_pos = chapter1_content.find("tyke")
            
            # Show the exact area around "Little tyke"
            start = max(0, little_pos - 50)
            end = min(len(chapter1_content), tyke_pos + 50)
            context = chapter1_content[start:end]
            
            print(f"  Context: {context}")
            
            # Check what characters are around "tyke"
            tyke_end = tyke_pos + 4  # "tyke" is 4 characters
            after_tyke = chapter1_content[tyke_end:tyke_end+10]
            print(f"  After 'tyke': {repr(after_tyke)}")
            
            # Look for any quote-like characters in this area
            quote_area = chapter1_content[little_pos-20:tyke_pos+20]
            print(f"  Quote area: {repr(quote_area)}")
            
            # Check for smart quotes specifically
            smart_quotes_left = quote_area.count('"')
            smart_quotes_right = quote_area.count('"')
            print(f"  Smart quotes in area: left={smart_quotes_left}, right={smart_quotes_right}")
        
        # Try to extract quotes with smart quotes
        print(f"\n🔍 Extracting quotes with smart quotes:")
        
        # Pattern for smart quotes
        smart_quote_pattern = r'"([^"]+)"'
        smart_quotes = re.findall(smart_quote_pattern, chapter1_content)
        print(f"  Found {len(smart_quotes)} smart-quoted segments")
        
        if smart_quotes:
            for i, quote in enumerate(smart_quotes[:5]):
                # Clean up the quote
                cleaned = re.sub(r'\s+', ' ', quote.strip())
                print(f"    {i+1}. {cleaned}")
            if len(smart_quotes) > 5:
                print(f"    ... and {len(smart_quotes) - 5} more")
        
        # Look for the exact "Little tyke" quote with smart quotes
        little_tyke_smart_pattern = r'"Little\s+tyke[^"]*"'
        matches = re.findall(little_tyke_smart_pattern, chapter1_content)
        print(f"\n🎯 Smart quote pattern for 'Little tyke': {len(matches)} matches")
        for i, match in enumerate(matches):
            print(f"    {i+1}. {match}")
        
        # Show all characters in the text (first 1000 chars)
        print(f"\n🔍 Character analysis (first 1000 chars):")
        sample = chapter1_content[:1000]
        for i, char in enumerate(sample):
            if char in ['"', '"', '"', "'", "'", "'", '—', '–', '-']:
                print(f"  Position {i}: '{char}' (ord: {ord(char)})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_smart_quotes() 