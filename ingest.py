import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Fix protobuf issue
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# -------------------------------
# Configuration
# -------------------------------

DATA_DIR = "./markdown"
VECTORSTORE_DIR = "./vectorstore"
DOC_SUMMARY_PATH = os.path.join(VECTORSTORE_DIR, "doc_summary.json")

EMBEDDING_MODEL = "all-mini_v12"

# -------------------------------
# Chunking Logic
# -------------------------------

def chunk_markdown(text, chunk_size=512, chunk_overlap=100):
    chunks = []
    lines = text.split("\n")
    
    current_chunk = ""
    for line in lines:
        if len(current_chunk) + len(line) < chunk_size:
            current_chunk += line + "\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            # start new chunk with overlap
            # A simple overlap strategy: take last N characters of the old chunk
            overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
            current_chunk = overlap_text + line + "\n"
            
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

# -------------------------------
# Embedding model setup
# -------------------------------

def get_embedding_model():
    model_name = os.environ.get("EMBEDDING_MODEL", EMBEDDING_MODEL)
    print(f"Loading embedding model: {model_name}")

    try:
        embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print(f"Embedding model loaded successfully")
        return embedder
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        raise

# -------------------------------
# Ingestion function
# -------------------------------

def ingest_all(
    data_dir=DATA_DIR, vector_dir=VECTORSTORE_DIR, chunk_size=512, chunk_overlap=100
):
    print("\n" + "=" * 60)
    print("Starting Markdown Ingestion Process")
    print("=" * 60 + "\n")

    if not os.path.exists(data_dir):
        print(f"Error: Directory not found: {data_dir}")
        return False

    try:
        embedder = get_embedding_model()
    except Exception as e:
        print(f"Failed to load embedding model: {e}")
        return False

    md_files = list(Path(data_dir).glob("*.md"))

    if not md_files:
        print(f"No MD files found in directory: {data_dir}")
        return False

    print(f"Found {len(md_files)} MD file(s) to process\n")

    docs = []
    doc_summaries = {}
    successful_mds = 0
    failed_mds = []

    for md in tqdm(md_files, desc="Processing MDs"):
        try:
            with open(md, "r", encoding="utf-8") as f:
                content = f.read()
            
            doc_id = md.stem
            chunks = chunk_markdown(content, chunk_size, chunk_overlap)
            
            if not chunks:
                print(f"Skipping (no content): {md.name}")
                failed_mds.append(md.name)
                continue

            chunk_docs = []
            for i, c_text in enumerate(chunks):
                metadata = {
                    "source": doc_id,
                    "chunk_id": i
                }
                chunk_docs.append(Document(page_content=c_text, metadata=metadata))

            docs.extend(chunk_docs)
            doc_summaries[doc_id] = {"num_chunks": len(chunks)}
            successful_mds += 1
            print(f"{md.name} - {len(chunks)} chunks")

        except Exception as e:
            print(f"Error processing {md.name}: {e}")
            failed_mds.append(md.name)

    if len(docs) == 0:
        print("\nNo valid chunks found in any MD.")
        return False

    print(f"\n{'='*60}")
    print(f"Processing Summary:")
    print(f"  Successfully processed: {successful_mds} MDs")
    print(f"  Failed: {len(failed_mds)} MDs")
    print(f"  Total chunks created: {len(docs)}")
    print(f"{'='*60}\n")

    try:
        print("Creating FAISS vector store...")
        os.makedirs(vector_dir, exist_ok=True)

        db = FAISS.from_documents(docs, embedder)
        db.save_local(vector_dir)
        print(f"FAISS index saved at: {vector_dir}")

    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        return False

    try:
        print("\nComputing document-level summaries...")
        chunk_texts = [d.page_content for d in docs]
        embeddings = embedder.embed_documents(chunk_texts)

        doc_vectors = {}
        for d, emb in zip(docs, embeddings):
            src = d.metadata.get("source", "unknown")
            doc_vectors.setdefault(src, []).append(np.array(emb, dtype=np.float32))

        doc_summary_store = {}
        for src, arrs in doc_vectors.items():
            mean_vec = np.mean(arrs, axis=0).astype(np.float32).tolist()
            doc_summary_store[src] = {
                "summary_vector": mean_vec,
                "num_chunks": len(arrs),
            }

        with open(DOC_SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(doc_summary_store, f, indent=2)

        print(f"Document-level summaries saved: {DOC_SUMMARY_PATH}")

    except Exception as e:
        print(f"Warning: Could not create document summaries: {e}")

    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print("=" * 60 + "\n")

    return True

if __name__ == "__main__":
    try:
        success = ingest_all()
        if success:
            print("All done! You can now run the FastAPI server (agents.py)")
        else:
            print("Ingestion failed. Please check the errors above.")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
