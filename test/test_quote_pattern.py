#!/usr/bin/env python
"""
Test script to scan the PDF for the specific quote '"Little tyke,"' and find quote patterns.
"""

import sys
import re
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

def test_quote_pattern():
    """Test to find the specific quote and identify quote patterns."""
    print("🔍 Scanning PDF for quote patterns...")
    
    # Path to the PDF
    pdf_path = "Harry Potter - Book 1 - The Sorcerers Stone.pdf"
    
    if not Path(pdf_path).exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    try:
        # Load the PDF
        print("📖 Loading PDF...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"✅ PDF loaded with {len(documents)} pages")
        
        # Search for the exact phrase '"Little tyke,"'
        target_quote = '"Little tyke,"'
        print(f"\n🎯 Searching for exact target quote: {target_quote}")
        found_target = False
        for i, doc in enumerate(documents):
            content = doc.page_content
            if not content.strip():
                continue
            if target_quote in content:
                found_target = True
                print(f"✅ Found target quote on page {i+1}")
                quote_index = content.find(target_quote)
                start = max(0, quote_index - 100)
                end = min(len(content), quote_index + len(target_quote) + 100)
                print(f"Context: ...{content[start:end]}...")
                break
        if not found_target:
            print(f"❌ Target quote {target_quote} not found in any page")
        
        # Now, print all double-quoted segments (including those with em-dashes inside)
        print(f"\n🔍 Scanning for all double-quoted segments (including those with em-dashes inside)...")
        double_quote_pattern = r'"[^"]+"'
        all_double_quotes = []
        for i, doc in enumerate(documents):
            content = doc.page_content
            if not content.strip():
                continue
            matches = re.findall(double_quote_pattern, content)
            if matches:
                for match in matches:
                    all_double_quotes.append((i+1, match))
        print(f"Found {len(all_double_quotes)} double-quoted segments in the document.")
        if all_double_quotes:
            print("\nSample double-quoted segments:")
            for idx, (page, quote) in enumerate(all_double_quotes[:10]):
                print(f"  Page {page}: {quote}")
        else:
            print("No double-quoted segments found.")
        
        # Additionally, print any double-quoted segments that contain em-dashes
        print(f"\n🔍 Double-quoted segments containing em-dashes:")
        em_dash_in_quote = [ (page, quote) for (page, quote) in all_double_quotes if '—' in quote ]
        if em_dash_in_quote:
            for idx, (page, quote) in enumerate(em_dash_in_quote[:10]):
                print(f"  Page {page}: {quote}")
        else:
            print("No double-quoted segments with em-dashes found.")
        
    except Exception as e:
        print(f"❌ Error during quote pattern search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_quote_pattern() 