# embeddings.py
import os, json
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


def embed_and_store(chunks_dir="chunks", persist_directory="vectordb", model_name="nomic-embed-text:latest", batch_size=50):
    print("🔄 Starting embedding process...")
    
    # Créer l'embedder
    embedder = OllamaEmbeddings(model=model_name)
    print("✅ Embedder initialized")

    # Charger tous les chunks
    documents = []
    chunk_files = [f for f in os.listdir(chunks_dir) if f.endswith(".json")]
    
    if not chunk_files:
        print("❌ No chunk files found!")
        return
    
    print(f"📂 Found {len(chunk_files)} chunk file(s)")
    
    for fn in chunk_files:
        print(f"   Loading {fn}...")
        with open(os.path.join(chunks_dir, fn), "r", encoding="utf8") as f:
            items = json.load(f)
        
        for it in items:
            doc = Document(
                page_content=it["page_content"],
                metadata=it.get("metadata", {})
            )
            documents.append(doc)
    
    print(f"📚 Total documents loaded: {len(documents)}")
    print(f"🔄 Creating embeddings in batches of {batch_size}...")
    print("   ⏳ This may take 5-15 minutes depending on document size...")

    # Créer la base vectorielle vide d'abord
    vectordb = None
    
    # Traiter par lots
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        print(f"   📦 Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
        
        try:
            if vectordb is None:
                # Première batch : créer la base
                vectordb = Chroma.from_documents(
                    documents=batch,
                    embedding=embedder,
                    persist_directory=persist_directory
                )
            else:
                # Batches suivantes : ajouter à la base existante
                vectordb.add_documents(batch)
            
            print(f"      ✅ Batch {batch_num} completed")
            
        except Exception as e:
            print(f"      ❌ Error in batch {batch_num}: {e}")
            print("      Retrying with smaller batch...")
            # Si erreur, réessayer avec des mini-batches de 10
            for j in range(0, len(batch), 10):
                mini_batch = batch[j:j + 10]
                try:
                    if vectordb is None:
                        vectordb = Chroma.from_documents(
                            documents=mini_batch,
                            embedding=embedder,
                            persist_directory=persist_directory
                        )
                    else:
                        vectordb.add_documents(mini_batch)
                    print(f"         ✅ Mini-batch {j//10 + 1} completed")
                except Exception as e2:
                    print(f"         ❌ Error in mini-batch: {e2}")
                    continue

    if vectordb is None:
        print("❌ Failed to create vector database!")
        return None

    print(f"\n✅ Successfully stored {len(documents)} vectors in {persist_directory}")
    
    # Test de la base de données
    print("\n🧪 Testing the vector database...")
    test_query = "sea level rise"
    try:
        test_results = vectordb.similarity_search(test_query, k=3)
        print(f"   Query: '{test_query}'")
        print(f"   Results found: {len(test_results)}")
        
        if test_results:
            print(f"\n   📄 First result:")
            print(f"      Source: {test_results[0].metadata.get('source', 'Unknown')}")
            print(f"      Page: {test_results[0].metadata.get('page', 'N/A')}")
            print(f"      Content preview: {test_results[0].page_content[:300]}...")
        else:
            print("   ⚠️ WARNING: No results found in test query!")
    except Exception as e:
        print(f"   ❌ Error during test: {e}")
    
    return vectordb


if __name__ == "__main__":
    embed_and_store()