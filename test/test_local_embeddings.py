#!/usr/bin/env python
"""
Quick test script to verify local embeddings work correctly.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

# Import the local embeddings function
from RAG_pipeline import create_local_embeddings

def test_local_embeddings():
    """Test that local embeddings can be created and used."""
    print("🧪 Testing local embeddings...")
    
    try:
        # Create local embeddings
        embeddings = create_local_embeddings()
        print("✅ Local embeddings created successfully")
        
        # Test embedding a simple text
        test_texts = [
            "Harry Potter is a wizard.",
            "The Sorcerer's Stone is a magical object.",
            "Hogwarts is a school for magic."
        ]
        
        print("🔢 Testing embedding generation...")
        embeddings_list = embeddings.embed_documents(test_texts)
        
        print(f"✅ Generated {len(embeddings_list)} embeddings")
        print(f"✅ Each embedding has {len(embeddings_list[0])} dimensions")
        
        # Test similarity
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        embeddings_array = np.array(embeddings_list)
        similarity_matrix = cosine_similarity(embeddings_array)
        
        print("📊 Similarity matrix:")
        print(similarity_matrix)
        
        print("✅ Local embeddings test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Local embeddings test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_local_embeddings()
    if success:
        print("\n🎉 Local embeddings are working correctly!")
        print("You can now run the full RAG pipeline without OpenAI quota issues.")
    else:
        print("\n❌ Local embeddings test failed. Check the error above.") 