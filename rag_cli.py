#!/usr/bin/env python3
"""
RAG Pipeline CLI - Command Line Interface for Harry Potter RAG System
"""

import argparse
import sys
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_pdf_exists():
    """Check if the Harry Potter PDF exists"""
    pdf_path = "Harry Potter - Book 1 - The Sorcerers Stone.pdf"
    if not Path(pdf_path).exists():
        logger.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)
    return pdf_path


def cmd_split_chapters(args):
    """Split PDF into chapters"""
    logger.info("🔪 Splitting PDF into chapters...")
    
    # Import here to avoid triggering main execution
    from helper_functions import split_into_chapters, replace_t_with_space
    
    pdf_path = check_pdf_exists()
    chapters = split_into_chapters(pdf_path)
    chapters = replace_t_with_space(chapters)
    
    logger.info(f"✅ Successfully split PDF into {len(chapters)} chapters")
    print(f"📊 Total chapters: {len(chapters)}")
    
    for i, chapter in enumerate(chapters, 1):
        print(f"  Chapter {i}: {len(chapter.page_content)} characters")


def cmd_extract_quotes(args):
    """Extract quotes from the PDF"""
    logger.info("💬 Extracting quotes from PDF...")
    
    # Import here to avoid triggering main execution
    from helper_functions import extract_book_quotes_as_documents, replace_t_with_space
    from langchain_community.document_loaders import PyPDFLoader
    
    pdf_path = check_pdf_exists()
    
    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    document_cleaned = replace_t_with_space(document)
    
    book_quotes_list = extract_book_quotes_as_documents(document_cleaned)
    
    logger.info(f"✅ Successfully extracted {len(book_quotes_list)} quotes")
    print(f"📊 Total quotes extracted: {len(book_quotes_list)}")
    
    if book_quotes_list:
        print("\n📝 Sample quotes:")
        for i, quote in enumerate(book_quotes_list[:5], 1):
            print(f"  {i}. {quote.page_content[:100]}...")


def cmd_status(args):
    """Show pipeline status"""
    logger.info("📊 Checking pipeline status...")
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found. Run ingestion first.")
        return
    
    print("=" * 50)
    print("📊 RAG PIPELINE STATUS")
    print("=" * 50)
    
    # Check PDF
    pdf_path = Path("Harry Potter - Book 1 - The Sorcerers Stone.pdf")
    if pdf_path.exists():
        print(f"✅ PDF: {pdf_path.name}")
    else:
        print(f"❌ PDF: {pdf_path.name} (missing)")
    
    # Check progress tracker
    progress_file = data_dir / "progress_tracker.json"
    if progress_file.exists():
        print("✅ Progress tracker: Found")
    else:
        print("❌ Progress tracker: Not found")
    
    # Check chapter summaries
    summaries_dir = data_dir / "chapter_summaries"
    if summaries_dir.exists():
        summary_files = list(summaries_dir.glob("*.json"))
        print(f"✅ Chapter summaries: {len(summary_files)} files")
    else:
        print("❌ Chapter summaries: Not found")
    
    # Check vector stores
    vector_stores = [
        ("Book chunks", data_dir / "chunks_vector_store"),
        ("Chapter summaries", data_dir / "chapter_summaries_vector_store"),
        ("Book quotes", data_dir / "book_quotes_vectorstore")
    ]
    
    for name, path in vector_stores:
        if path.exists() and (path / "index.faiss").exists():
            print(f"✅ {name} vector store: Ready")
        else:
            print(f"❌ {name} vector store: Not found")


def cmd_qa(args):
    """Start interactive Q&A session"""
    logger.info("🎭 Starting interactive Q&A session...")
    
    # Import here to avoid triggering main execution
    from RAG_pipeline import interactive_qa
    interactive_qa()


def cmd_ask(args):
    """Ask a single question"""
    if not args.question:
        print("❌ Please provide a question with --question")
        return
    
    logger.info(f"🤔 Processing question: {args.question}")
    
    try:
        # Import here to avoid triggering main execution
        from RAG_pipeline import execute_plan_and_print_steps
        
        input_data = {"question": args.question}
        final_answer, final_state = execute_plan_and_print_steps(input_data)
        
        print("\n" + "=" * 60)
        print("🎯 ANSWER:")
        print("=" * 60)
        print(final_answer)
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        print(f"❌ Error: {e}")


def cmd_create_summaries(args):
    """Create chapter summaries"""
    logger.info("📝 Creating chapter summaries...")
    
    # Import here to avoid triggering main execution
    from RAG_pipeline import (
        setup_progress_tracking,
        load_progress,
        save_progress,
        resume_chapter_processing
    )
    
    pdf_path = check_pdf_exists()
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Setup paths
    progress_file_path = data_dir / "progress_tracker.json"
    summaries_dir_path = data_dir / "chapter_summaries"
    
    # Create summaries directory
    summaries_dir_path.mkdir(exist_ok=True)
    
    # Split chapters
    from helper_functions import split_into_chapters, replace_t_with_space
    chapters = split_into_chapters(pdf_path)
    chapters = replace_t_with_space(chapters)
    
    # Setup progress tracking
    progress = setup_progress_tracking()
    save_progress(progress_file_path, progress)
    
    # Resume processing (this will handle all chapters)
    resume_chapter_processing(chapters, progress_file_path, summaries_dir_path)
    
    logger.info("✅ Chapter summaries completed")
    print("📊 Check data/chapter_summaries/ for individual summary files")


def cmd_create_embeddings(args):
    """Create vector embeddings for chunks, summaries, and quotes"""
    logger.info("🧠 Creating vector embeddings...")
    
    # Import here to avoid triggering main execution
    from RAG_pipeline import (
        create_local_embeddings,
        encode_book,
        encode_chapter_summaries,
        encode_quotes,
        load_chapter_summary
    )
    
    pdf_path = check_pdf_exists()
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 1. Create embeddings for book chunks
    logger.info("📚 Creating embeddings for book chunks...")
    encode_book(pdf_path)
    print("✅ Book chunks embeddings created")
    
    # 2. Create embeddings for chapter summaries
    logger.info("📝 Creating embeddings for chapter summaries...")
    summaries_dir = data_dir / "chapter_summaries"
    if summaries_dir.exists():
        chapter_summaries = []
        for i in range(1, 18):  # Assuming 17 chapters
            summary = load_chapter_summary(summaries_dir, i)
            if summary:
                chapter_summaries.append(summary)
        
        if chapter_summaries:
            encode_chapter_summaries(chapter_summaries)
            print(f"✅ Chapter summaries embeddings created ({len(chapter_summaries)} summaries)")
        else:
            print("⚠️  No chapter summaries found. Run 'create-summaries' first.")
    else:
        print("⚠️  Chapter summaries directory not found. Run 'create-summaries' first.")
    
    # 3. Create embeddings for quotes
    logger.info("💬 Creating embeddings for book quotes...")
    from helper_functions import extract_book_quotes_as_documents, replace_t_with_space
    from langchain_community.document_loaders import PyPDFLoader
    
    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    document_cleaned = replace_t_with_space(document)
    book_quotes_list = extract_book_quotes_as_documents(document_cleaned)
    
    if book_quotes_list:
        encode_quotes(book_quotes_list)
        print(f"✅ Book quotes embeddings created ({len(book_quotes_list)} quotes)")
    else:
        print("⚠️  No quotes found in the document")
    
    logger.info("✅ All embeddings completed")


def cmd_ingest_all(args):
    """Run complete ingestion pipeline"""
    logger.info("🚀 Running complete ingestion pipeline...")
    
    print("=" * 60)
    print("📚 HARRY POTTER RAG - COMPLETE INGESTION PIPELINE")
    print("=" * 60)
    
    # Step 1: Split chapters
    print("\n1️⃣  Splitting PDF into chapters...")
    cmd_split_chapters(args)
    
    # Step 2: Extract quotes
    print("\n2️⃣  Extracting quotes...")
    cmd_extract_quotes(args)
    
    # Step 3: Create summaries
    print("\n3️⃣  Creating chapter summaries...")
    cmd_create_summaries(args)
    
    # Step 4: Create embeddings
    print("\n4️⃣  Creating vector embeddings...")
    cmd_create_embeddings(args)
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE INGESTION PIPELINE FINISHED!")
    print("=" * 60)
    print("📁 Data stored in: data/")
    print("🔍 Ready for Q&A! Run: uv run python rag_cli.py qa")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Harry Potter RAG Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete ingestion pipeline
  uv run python rag_cli.py ingest-all
  
  # Run individual stages
  uv run python rag_cli.py split-chapters
  uv run python rag_cli.py extract-quotes
  uv run python rag_cli.py create-summaries
  uv run python rag_cli.py create-embeddings
  
  # Check pipeline status
  uv run python rag_cli.py status
  
  # Interactive Q&A
  uv run python rag_cli.py qa
  
  # Ask a single question
  uv run python rag_cli.py ask --question "What is Harry's owl's name?"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Split chapters command
    split_parser = subparsers.add_parser('split-chapters', help='Split PDF into chapters')
    
    # Extract quotes command
    quotes_parser = subparsers.add_parser('extract-quotes', help='Extract quotes from PDF')
    
    # Create summaries command
    summaries_parser = subparsers.add_parser('create-summaries', help='Create chapter summaries')
    
    # Create embeddings command
    embeddings_parser = subparsers.add_parser('create-embeddings', help='Create vector embeddings')
    
    # Ingest all command
    ingest_parser = subparsers.add_parser('ingest-all', help='Run complete ingestion pipeline')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show pipeline status')
    
    # Q&A command
    qa_parser = subparsers.add_parser('qa', help='Start interactive Q&A session')
    
    # Ask command
    ask_parser = subparsers.add_parser('ask', help='Ask a single question')
    ask_parser.add_argument('--question', '-q', required=True, help='Question to ask')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute the appropriate command
    command_handlers = {
        'split-chapters': cmd_split_chapters,
        'extract-quotes': cmd_extract_quotes,
        'status': cmd_status,
        'qa': cmd_qa,
        'ask': cmd_ask,
        'create-summaries': cmd_create_summaries,
        'create-embeddings': cmd_create_embeddings,
        'ingest-all': cmd_ingest_all
    }
    
    try:
        command_handlers[args.command](args)
    except KeyboardInterrupt:
        print("\n\n👋 Operation interrupted by user")
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
