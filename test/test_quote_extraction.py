#!/usr/bin/env python
"""
Test script to extract quotes from the Harry Potter PDF and report results.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

# Import the quote extraction function
from helper_functions import extract_book_quotes_as_documents
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

def test_quote_extraction():
    """Test quote extraction from the Harry Potter PDF."""
    print("🧪 Testing quote extraction from Harry Potter PDF...")
    
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
        
        # Check content of first few pages
        print("\n🔍 Checking content of first few pages...")
        for i in range(min(5, len(documents))):
            content = documents[i].page_content
            print(f"Page {i+1}: {len(content)} characters")
            if content.strip():
                print(f"  Sample: '{content[:100]}{'...' if len(content) > 100 else ''}'")
            else:
                print(f"  Empty content")
        
        # Clean the documents (replace tabs with spaces)
        print("\n🧹 Cleaning documents...")
        cleaned_documents = []
        for doc in documents:
            cleaned_content = doc.page_content.replace('\t', ' ')
            cleaned_doc = Document(page_content=cleaned_content, metadata=doc.metadata)
            cleaned_documents.append(cleaned_doc)
        print("✅ Documents cleaned")
        
        # Extract quotes with different minimum lengths
        print("\n🔍 Extracting quotes...")
        
        # Test with default min_length=50
        quotes_50 = extract_book_quotes_as_documents(cleaned_documents, min_length=50)
        print(f"📊 Quotes with min_length=50: {len(quotes_50)}")
        
        # Test with min_length=20
        quotes_20 = extract_book_quotes_as_documents(cleaned_documents, min_length=20)
        print(f"📊 Quotes with min_length=20: {len(quotes_20)}")
        
        # Test with min_length=10
        quotes_10 = extract_book_quotes_as_documents(cleaned_documents, min_length=10)
        print(f"📊 Quotes with min_length=10: {len(quotes_10)}")
        
        # Test with min_length=5
        quotes_5 = extract_book_quotes_as_documents(cleaned_documents, min_length=5)
        print(f"📊 Quotes with min_length=5: {len(quotes_5)}")
        
        # Show some sample quotes if any found
        if quotes_5:
            print(f"\n📝 Sample quotes (first 5):")
            for i, quote in enumerate(quotes_5[:5]):
                print(f"  {i+1}. \"{quote.page_content[:100]}{'...' if len(quote.page_content) > 100 else ''}\"")
        else:
            print("\n❌ No quotes found with any minimum length!")
            
            # Let's check what the text looks like in pages with content
            print("\n🔍 Analyzing text content...")
            for i, doc in enumerate(cleaned_documents[:10]):
                if doc.page_content.strip():
                    sample_text = doc.page_content[:500]
                    print(f"\nPage {i+1} content (first 500 chars):")
                    print(f"'{sample_text}'")
                    
                    # Check for quote patterns
                    import re
                    quote_pattern = re.compile(r'"[^"]*"')
                    all_quotes = quote_pattern.findall(sample_text)
                    print(f"Found {len(all_quotes)} quote patterns in page {i+1}:")
                    for j, quote in enumerate(all_quotes[:5]):
                        print(f"  {j+1}. {quote}")
                    break  # Just check the first page with content
        
    except Exception as e:
        print(f"❌ Error during quote extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_quote_extraction() 