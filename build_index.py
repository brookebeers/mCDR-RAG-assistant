import os
import csv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

DOCS_FOLDER = "formatted_docs"
METADATA_CSV = "lit-tag-database.csv"
PINECONE_INDEX_NAME = "ices-database-assistant"

# load API key from environment variable/secrets
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("Missing Pinecone API key. Set PINECONE_API_KEY as an environment variable.")

model = SentenceTransformer("all-MiniLM-L6-v2")
pc = Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1")
index = pc.Index(PINECONE_INDEX_NAME)

# functions
def chunk_text(text, chunk_size=300):
    """Split text into word-based chunks."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def load_metadata():
    """Load metadata from CSV into a lookup dict keyed by filename."""
    metadata_lookup = {}
    with open(METADATA_CSV, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            doc_id = row.get("key")
            metadata_lookup[f"{doc_id}.txt"] = row
    return metadata_lookup

def prepare_documents():
    """Read documents, chunk them, and attach metadata."""
    metadata_lookup = load_metadata()
    documents, metadata_list = [], []

    for filename in os.listdir(DOCS_FOLDER):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(DOCS_FOLDER, filename), "r", encoding="utf-8") as f:
            text = f.read()
            chunks = chunk_text(text)
            meta = metadata_lookup.get(filename, {})
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadata_list.append({
                    "filename": filename,
                    "chunk_id": i,
                    "text": chunk,
                    **meta
                })
    return documents, metadata_list

def batch_upsert(index, vectors, batch_size=100):
    """Upload vectors to Pinecone in batches."""
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i+batch_size])

# main workflow
if __name__ == "__main__":
    print("Preparing documents...")
    documents, metadata_list = prepare_documents()

    print(f"Encoding {len(documents)} chunks...")
    embeddings = model.encode(documents, show_progress_bar=True)

    print("Building vectors...")
    vectors = [
        (f"{meta['filename']}_chunk_{meta['chunk_id']}", embeddings[i].tolist(), meta)
        for i, meta in enumerate(metadata_list)
    ]

    print(f"Upserting {len(vectors)} vectors into Pinecone index '{PINECONE_INDEX_NAME}'...")
    batch_upsert(index, vectors)
    print("Index build complete")
