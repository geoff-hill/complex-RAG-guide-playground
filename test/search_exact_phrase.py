#!/usr/bin/env python
"""
Search for the exact phrase "Little tyke," in Chapter 1 content.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from helper_functions import split_into_chapters

def search_exact_phrase():
    """Search for the exact phrase 'Little tyke,' in Chapter 1."""
    print("🔍 Searching for exact phrase 'Little tyke,' in Chapter 1...")
    
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
        
        # Search for the exact phrase
        target_phrase = "Little tyke,"
        
        print(f"📖 Searching for: '{target_phrase}'")
        print(f"📊 Content length: {len(chapter1_content)} characters")
        
        # Method 1: Simple string search
        if target_phrase in chapter1_content:
            print("✅ Found with simple string search!")
            # Find all occurrences
            start_pos = chapter1_content.find(target_phrase)
            print(f"   First occurrence at position: {start_pos}")
            
            # Show context around the first occurrence
            context_start = max(0, start_pos - 100)
            context_end = min(len(chapter1_content), start_pos + len(target_phrase) + 100)
            context = chapter1_content[context_start:context_end]
            print(f"   Context: ...{context}...")
            
        else:
            print("❌ Not found with simple string search")
        
        # Method 2: Case insensitive search
        if target_phrase.lower() in chapter1_content.lower():
            print("✅ Found with case insensitive search!")
        else:
            print("❌ Not found with case insensitive search")
        
        # Method 3: Search for variations
        variations = [
            "Little tyke,",
            "Little tyke",
            "little tyke,",
            "little tyke",
            '"Little tyke,"',
            '"Little tyke"',
            "'Little tyke,'",
            "'Little tyke'"
        ]
        
        print(f"\n🔍 Testing variations:")
        for variation in variations:
            if variation in chapter1_content:
                print(f"  ✅ Found: '{variation}'")
                # Show context
                start_pos = chapter1_content.find(variation)
                context_start = max(0, start_pos - 50)
                context_end = min(len(chapter1_content), start_pos + len(variation) + 50)
                context = chapter1_content[context_start:context_end]
                print(f"     Context: ...{context}...")
            else:
                print(f"  ❌ Not found: '{variation}'")
        
        # Method 4: Look for the words separately
        print(f"\n🔍 Looking for words separately:")
        if "Little" in chapter1_content and "tyke" in chapter1_content:
            print("  ✅ Both 'Little' and 'tyke' found separately")
            # Find positions
            little_pos = chapter1_content.find("Little")
            tyke_pos = chapter1_content.find("tyke")
            print(f"    'Little' at position: {little_pos}")
            print(f"    'tyke' at position: {tyke_pos}")
            
            # Check if they're close together
            if abs(little_pos - tyke_pos) < 20:
                print(f"    They appear close together (distance: {abs(little_pos - tyke_pos)})")
                # Show context around both
                start = min(little_pos, tyke_pos)
                end = max(little_pos, tyke_pos) + 10
                context = chapter1_content[start:end]
                print(f"    Context: {context}")
        else:
            print("  ❌ One or both words not found")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    search_exact_phrase() 