import os, json, yaml
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

CFG = yaml.safe_load(open("config.yaml"))

chunks_path = CFG["corpus_chunks_path"]
os.makedirs(CFG["artifacts_dir"], exist_ok=True)

# Use ChromaDB instead of FAISS
chroma_client = chromadb.PersistentClient(
    path=os.path.join(CFG["artifacts_dir"], "chroma_db"),
    settings=Settings(anonymized_telemetry=False)
)

# Load chunks
chunks = [json.loads(l) for l in open(chunks_path) if l.strip()]
texts = [c["text"] for c in chunks]
metadatas = [c.get("meta", {}) for c in chunks]
ids = [f"chunk_{i}" for i in range(len(chunks))]

print(f"Loaded {len(texts)} chunks from {chunks_path}")

# Create collection
collection = chroma_client.create_collection(
    name="finance_corpus",
    metadata={"description": "Finance RAG corpus chunks"}
)

# Add documents to collection
collection.add(
    documents=texts,
    metadatas=metadatas,
    ids=ids
)

print(f"Saved ChromaDB collection with {len(texts)} documents")
print("ChromaDB path:", os.path.join(CFG["artifacts_dir"], "chroma_db"))
