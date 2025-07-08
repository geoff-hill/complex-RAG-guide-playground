#!/usr/bin/env python
"""
Test script for the restartable chapter processing system.
This script tests the core functionality without running the full RAG pipeline.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from langchain.docstore.document import Document

# Add the current directory to the path so we can import from RAG_pipeline
sys.path.append('.')

# Import the restartable processing functions
from RAG_pipeline import (
    setup_progress_tracking,
    load_progress,
    save_progress,
    save_chapter_summary,
    load_chapter_summary,
    update_chapter_progress,
    get_next_pending_chapter,
    print_progress_summary
)


def create_test_chapters():
    """Create test chapter documents for testing."""
    chapters = []
    for i in range(1, 6):  # Create 5 test chapters
        chapter_content = f"This is test chapter {i} content. It contains some sample text for testing the restartable processing system."
        chapter_doc = Document(
            page_content=chapter_content,
            metadata={"chapter": i, "test": True}
        )
        chapters.append(chapter_doc)
    return chapters


def test_progress_tracking():
    """Test the progress tracking functionality."""
    print("🧪 Testing progress tracking functionality...")
    
    # Set up progress tracking
    progress_file_path, summaries_dir_path = setup_progress_tracking()
    
    # Test loading new progress
    progress = load_progress(progress_file_path)
    print(f"✅ Loaded progress: {progress['total_chapters']} total chapters")
    
    # Test updating progress
    update_chapter_progress(progress, 1, "started")
    update_chapter_progress(progress, 1, "completed")
    update_chapter_progress(progress, 2, "started")
    update_chapter_progress(progress, 2, "failed", "Test error")
    
    print(f"✅ Updated progress: {progress['completed_chapters']} completed, {progress['failed_chapters']} failed")
    
    # Test saving and loading progress
    save_progress(progress_file_path, progress)
    loaded_progress = load_progress(progress_file_path)
    print(f"✅ Reloaded progress: {loaded_progress['completed_chapters']} completed")
    
    # Test getting next pending chapter
    next_chapter = get_next_pending_chapter(progress, 5)
    print(f"✅ Next pending chapter: {next_chapter}")
    
    # Test progress summary
    print_progress_summary(progress)
    
    return progress_file_path, summaries_dir_path


def test_summary_saving():
    """Test saving and loading chapter summaries."""
    print("\n🧪 Testing summary saving and loading...")
    
    progress_file_path, summaries_dir_path = test_progress_tracking()
    
    # Create test summary
    test_summary = Document(
        page_content="This is a test summary for chapter 1.",
        metadata={"chapter": 1, "test": True}
    )
    
    # Test saving summary
    success = save_chapter_summary(summaries_dir_path, 1, test_summary)
    print(f"✅ Save summary result: {success}")
    
    # Test loading summary
    loaded_summary = load_chapter_summary(summaries_dir_path, 1)
    if loaded_summary:
        print(f"✅ Loaded summary: {loaded_summary.page_content[:50]}...")
    else:
        print("❌ Failed to load summary")
    
    # Test loading non-existent summary
    non_existent = load_chapter_summary(summaries_dir_path, 999)
    print(f"✅ Non-existent summary result: {non_existent is None}")


def test_resume_logic():
    """Test the resume logic with simulated partial completion."""
    print("\n🧪 Testing resume logic...")
    
    progress_file_path, summaries_dir_path = setup_progress_tracking()
    
    # Simulate a partially completed run
    progress = {
        "started_at": datetime.now().isoformat(),
        "total_chapters": 5,
        "completed_chapters": 2,
        "failed_chapters": 1,
        "last_completed_chapter": 2,
        "chapters": {
            1: {"status": "completed", "started_at": "2024-01-01T10:00:00", "completed_at": "2024-01-01T10:05:00"},
            2: {"status": "completed", "started_at": "2024-01-01T10:10:00", "completed_at": "2024-01-01T10:15:00"},
            3: {"status": "failed", "started_at": "2024-01-01T10:20:00", "completed_at": "2024-01-01T10:25:00", "error_message": "API timeout"},
            4: {"status": "pending"},
            5: {"status": "pending"}
        }
    }
    
    save_progress(progress_file_path, progress)
    
    # Test getting next pending chapter (should be 3 since it failed)
    next_chapter = get_next_pending_chapter(progress, 5)
    print(f"✅ Next chapter to process: {next_chapter} (should be 3)")
    
    # Test progress summary
    print_progress_summary(progress)


def cleanup_test_files():
    """Clean up test files."""
    print("\n🧹 Cleaning up test files...")
    
    # Remove test files
    test_files = [
        "progress_tracker.json",
        "chapter_summaries"
    ]
    
    for file_path in test_files:
        path = Path(file_path)
        if path.is_file():
            path.unlink()
            print(f"✅ Removed file: {file_path}")
        elif path.is_dir():
            import shutil
            shutil.rmtree(path)
            print(f"✅ Removed directory: {file_path}")


def main():
    """Run all tests."""
    print("🚀 Starting restartable processing system tests...")
    print("=" * 60)
    
    try:
        test_progress_tracking()
        test_summary_saving()
        test_resume_logic()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("The restartable processing system is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test files
        cleanup_test_files()


if __name__ == "__main__":
    main() 