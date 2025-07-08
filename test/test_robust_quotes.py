#!/usr/bin/env python
"""
Test robust quote extraction that handles unusual PDF spacing.
"""

import sys
import re
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from helper_functions import split_into_chapters

def extract_quotes_robust(text, min_length=10):
    """
    Extract quotes from text that may have unusual spacing.
    Handles various quote patterns and spacing issues.
    """
    quotes = []
    
    # Pattern 1: Standard double quotes with flexible spacing
    # This handles cases like "Little     tyke," where there are extra spaces
    pattern1 = r'"([^"]{10,})"'
    matches1 = re.findall(pattern1, text)
    quotes.extend(matches1)
    
    # Pattern 2: Look for dialogue with em-dashes (common in this book)
    # Find text between em-dashes that looks like dialogue
    pattern2 = r'—\s*([^—]{10,})\s*—'
    matches2 = re.findall(pattern2, text)
    quotes.extend(matches2)
    
    # Pattern 3: Look for dialogue patterns with em-dash at start
    pattern3 = r'—\s*([^—\n]{10,})'
    matches3 = re.findall(pattern3, text)
    quotes.extend(matches3)
    
    # Pattern 4: Look for dialogue patterns with em-dash at end
    pattern4 = r'([^—\n]{10,})\s*—'
    matches4 = re.findall(pattern4, text)
    quotes.extend(matches4)
    
    # Clean up quotes
    cleaned_quotes = []
    for quote in quotes:
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', quote.strip())
        if len(cleaned) >= min_length:
            cleaned_quotes.append(cleaned)
    
    return list(set(cleaned_quotes))  # Remove duplicates

def test_robust_quotes():
    """Test robust quote extraction on Chapter 1."""
    print("🔍 Testing robust quote extraction on Chapter 1...")
    
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
        
        # Extract quotes with different minimum lengths
        for min_length in [5, 10, 20, 30]:
            print(f"\n🔍 Extracting quotes with minimum length {min_length}:")
            quotes = extract_quotes_robust(chapter1_content, min_length)
            print(f"  Found {len(quotes)} quotes")
            
            if quotes:
                print("  Sample quotes:")
                for i, quote in enumerate(quotes[:5]):
                    print(f"    {i+1}. {quote}")
                if len(quotes) > 5:
                    print(f"    ... and {len(quotes) - 5} more")
        
        # Specifically look for "Little tyke" pattern
        print(f"\n🎯 Looking specifically for 'Little tyke' pattern:")
        
        # Search for the pattern with flexible spacing
        little_tyke_pattern = r'Little\s+tyke[^"]*"'
        matches = re.findall(little_tyke_pattern, chapter1_content)
        print(f"  Found {len(matches)} matches for 'Little tyke' pattern")
        for i, match in enumerate(matches):
            print(f"    {i+1}. {match}")
        
        # Look for the exact context around "Little tyke"
        if "Little" in chapter1_content and "tyke" in chapter1_content:
            little_pos = chapter1_content.find("Little")
            tyke_pos = chapter1_content.find("tyke")
            
            # Show more context around this area
            start = max(0, little_pos - 200)
            end = min(len(chapter1_content), tyke_pos + 200)
            context = chapter1_content[start:end]
            print(f"\n📖 Context around 'Little tyke':")
            print(f"  {context}")
            
            # Try to extract the quote from this context
            quote_match = re.search(r'"([^"]+)"', context)
            if quote_match:
                print(f"\n✅ Found quote: '{quote_match.group(1)}'")
            else:
                print(f"\n❌ No quote found in context")
        
        # Look for all double-quoted segments
        print(f"\n🔍 All double-quoted segments in Chapter 1:")
        all_quotes = re.findall(r'"([^"]+)"', chapter1_content)
        print(f"  Found {len(all_quotes)} double-quoted segments")
        
        if all_quotes:
            for i, quote in enumerate(all_quotes[:10]):
                # Clean up the quote
                cleaned = re.sub(r'\s+', ' ', quote.strip())
                print(f"    {i+1}. {cleaned}")
            if len(all_quotes) > 10:
                print(f"    ... and {len(all_quotes) - 10} more")
        else:
            print("  No double-quoted segments found")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_robust_quotes() 