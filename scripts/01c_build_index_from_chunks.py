import os, json, yaml
from sentence_transformers import SentenceTransformer
import faiss

CFG = yaml.safe_load(open("config.yaml"))

chunks_path = CFG["corpus_chunks_path"]
index_path  = CFG["faiss_index_path"]
os.makedirs(CFG["artifacts_dir"], exist_ok=True)

chunks = [json.loads(l) for l in open(chunks_path) if l.strip()]
texts  = [c["text"] for c in chunks]

print(f"Loaded {len(texts)} chunks from {chunks_path}")

embedder = SentenceTransformer(CFG["embedder_name"])
vecs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)

index = faiss.IndexFlatIP(vecs.shape[1])  # cosine via normalized inner product
index.add(vecs)

faiss.write_index(index, index_path)
print("Saved FAISS index ->", index_path)
