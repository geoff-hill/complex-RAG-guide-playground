#!/usr/bin/env python
"""
Check the exact character codes around the "Little tyke" quote.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from helper_functions import split_into_chapters

def check_character_codes():
    """Check the exact character codes around the 'Little tyke' quote."""
    print("🔍 Checking character codes around 'Little tyke' quote...")
    
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
        
        # Find "Little tyke"
        if "Little" in chapter1_content and "tyke" in chapter1_content:
            little_pos = chapter1_content.find("Little")
            tyke_pos = chapter1_content.find("tyke")
            
            print(f"📍 'Little' found at position: {little_pos}")
            print(f"📍 'tyke' found at position: {tyke_pos}")
            
            # Look at characters around "Little tyke"
            start = max(0, little_pos - 10)
            end = min(len(chapter1_content), tyke_pos + 15)
            area = chapter1_content[start:end]
            
            print(f"\n📖 Text area around 'Little tyke':")
            print(f"  {area}")
            
            print(f"\n🔍 Character-by-character analysis:")
            for i, char in enumerate(area):
                pos = start + i
                print(f"  Position {pos}: '{char}' (ord: {ord(char)}, hex: {ord(char):02x})")
            
            # Look specifically for quote characters
            print(f"\n🎯 Quote characters in the area:")
            for i, char in enumerate(area):
                if char in ['"', '"', '"', "'", "'", "'"]:
                    pos = start + i
                    print(f"  Position {pos}: '{char}' (ord: {ord(char)}, hex: {ord(char):02x})")
            
            # Check if there are any non-printable characters
            print(f"\n🔍 Non-printable characters in the area:")
            for i, char in enumerate(area):
                if not char.isprintable() and char != '\n' and char != '\t':
                    pos = start + i
                    print(f"  Position {pos}: '{char}' (ord: {ord(char)}, hex: {ord(char):02x})")
            
            # Try to extract the quote manually
            print(f"\n🎯 Manual quote extraction:")
            
            # Look for the opening quote before "Little"
            for i in range(little_pos - 1, max(0, little_pos - 20), -1):
                char = chapter1_content[i]
                if char in ['"', '"', '"']:
                    print(f"  Opening quote found at position {i}: '{char}' (ord: {ord(char)})")
                    break
                elif char.isspace():
                    continue
                else:
                    print(f"  No opening quote found before 'Little'")
                    break
            
            # Look for the closing quote after "tyke"
            tyke_end = tyke_pos + 4  # "tyke" is 4 characters
            for i in range(tyke_end, min(len(chapter1_content), tyke_end + 20)):
                char = chapter1_content[i]
                if char in ['"', '"', '"']:
                    print(f"  Closing quote found at position {i}: '{char}' (ord: {ord(char)})")
                    break
                elif char.isspace():
                    continue
                else:
                    print(f"  No closing quote found after 'tyke'")
                    break
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_character_codes() 