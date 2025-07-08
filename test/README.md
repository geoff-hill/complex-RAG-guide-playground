# Test Scripts

This directory contains various test scripts for the RAG pipeline.

## How to Run Tests

All tests should be run from the **root directory** of the project using `uv run`:

```bash
# Run tests from root directory using uv
uv run python test/test_script_name.py
```

## Available Tests

### Quote Extraction Tests
- `test_updated_quotes.py` - Tests the updated quote extraction with smart quotes
- `test_robust_quotes.py` - Tests robust quote extraction with various patterns
- `test_smart_quotes.py` - Tests for smart quotes and character analysis
- `test_quote_extraction.py` - Tests basic quote extraction functionality
- `test_quote_pattern.py` - Tests quote pattern matching

### Content Analysis Tests
- `extract_chapter1.py` - Extracts and displays Chapter 1 content
- `search_exact_phrase.py` - Searches for specific phrases in Chapter 1
- `check_character_codes.py` - Analyzes character codes around specific text

### Embedding Tests
- `test_embeddings_only.py` - Tests embedding functionality
- `test_local_embeddings.py` - Tests local embedding models

### Pipeline Tests
- `test_restartable.py` - Tests the restartable chapter processing functionality

### General Tests
- `test.py` - General test script

## Test Results

The tests verify:
- ✅ Smart quote extraction (Unicode 8220/8221)
- ✅ Text spacing cleanup
- ✅ Quote pattern matching
- ✅ Chapter content extraction
- ✅ Local embedding functionality
- ✅ Progress tracking
- ✅ Vector store operations

## Notes

- All tests use the Harry Potter PDF file from the root directory
- Tests import from `helper_functions.py` in the root directory
- Use `uv run` to automatically handle the virtual environment and dependencies 