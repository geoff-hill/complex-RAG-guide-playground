#!/usr/bin/env python
"""
Simple test script to verify local embeddings work correctly.
"""

from langchain_huggingface import HuggingFaceEmbeddings

def test_local_embeddings():
    """Test that local embeddings can be created and used."""
    print("🧪 Testing local embeddings...")
    
    try:
        # Create local embeddings
        print("🤖 Creating local embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
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