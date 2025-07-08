#!/usr/bin/env python
"""
Extract and display Chapter 1 from the Harry Potter PDF.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from helper_functions import split_into_chapters

def extract_chapter1():
    """Extract and display Chapter 1 from the Harry Potter PDF."""
    print("📖 Extracting Chapter 1 from Harry Potter PDF...")
    
    # Path to the PDF
    pdf_path = "Harry Potter - Book 1 - The Sorcerers Stone.pdf"
    
    if not Path(pdf_path).exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    try:
        # Split the PDF into chapters
        print("🔪 Splitting PDF into chapters...")
        chapters = split_into_chapters(pdf_path)
        print(f"✅ Successfully split PDF into {len(chapters)} chapters")
        
        if len(chapters) == 0:
            print("❌ No chapters found")
            return
        
        # Get Chapter 1 (index 0)
        chapter1 = chapters[0]
        print(f"\n📄 Chapter 1 Content:")
        print("=" * 80)
        print(f"Chapter: {chapter1.metadata.get('chapter', 'Unknown')}")
        print("=" * 80)
        
        # Display the content
        content = chapter1.page_content
        print(content)
        print("=" * 80)
        
        # Also show some statistics
        print(f"\n📊 Chapter 1 Statistics:")
        print(f"  Total characters: {len(content)}")
        print(f"  Total words: {len(content.split())}")
        print(f"  Lines: {len(content.split(chr(10)))}")
        
        # Check for various quote patterns
        import re
        
        # Check for different quote types
        quote_patterns = {
            'Double quotes': r'"[^"]*"',
            'Single quotes': r"'[^']*'",
            'Smart quotes left': r'"[^"]*"',
            'Smart quotes right': r'"[^"]*"',
            'Em dashes': r'—',
            'Regular dashes': r'-',
        }
        
        print(f"\n🔍 Quote Pattern Analysis:")
        for pattern_name, pattern in quote_patterns.items():
            matches = re.findall(pattern, content)
            print(f"  {pattern_name}: {len(matches)} found")
            if matches and len(matches) <= 5:
                for i, match in enumerate(matches[:3]):
                    print(f"    {i+1}. {match}")
        
        # Search specifically for "Little tyke"
        print(f"\n🎯 Searching for 'Little tyke':")
        if 'Little tyke' in content:
            print("  ✅ Found 'Little tyke' in Chapter 1")
            # Find all occurrences
            import re
            matches = re.finditer(r'Little tyke', content, re.IGNORECASE)
            for i, match in enumerate(matches):
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                context = content[start:end]
                print(f"    Occurrence {i+1}: ...{context}...")
        else:
            print("  ❌ 'Little tyke' not found in Chapter 1")
        
    except Exception as e:
        print(f"❌ Error extracting Chapter 1: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract_chapter1() 