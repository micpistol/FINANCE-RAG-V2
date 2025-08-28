import json, yaml, sys
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import chromadb
from chromadb.config import Settings
import os

CFG = yaml.safe_load(open("config.yaml"))

# Load retriever pieces (ChromaDB instead of FAISS)
embedder = SentenceTransformer(CFG["embedder_name"])
chroma_client = chromadb.PersistentClient(
    path=os.path.join(CFG["artifacts_dir"], "chroma_db"),
    settings=Settings(anonymized_telemetry=False)
)
collection = chroma_client.get_collection("finance_corpus")

# Load model
tok = AutoTokenizer.from_pretrained(CFG["finetuned_dir"])
model = AutoModelForSeq2SeqLM.from_pretrained(CFG["finetuned_dir"])

def retrieve(question, k=4):
    # Query ChromaDB
    results = collection.query(
        query_texts=[question],
        n_results=k
    )
    
    ctx_texts, ctx_citations = [], []
    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        metadata = results['metadatas'][0][i] or {}
        
        citation = {
            "source": metadata.get("source") or "unknown",
            "section": metadata.get("section"),
        }
        ctx_texts.append(text)
        ctx_citations.append(citation)
    
    return "\n---\n".join(ctx_texts), ctx_citations

def answer(question):
    context, cites = retrieve(question, CFG["retriever_top_k"])
    prompt_in = f"instruction: {question}\ncontext: {context}"
    enc = tok(prompt_in, return_tensors="pt", truncation=True, max_length=CFG["max_source_length"])
    out = model.generate(**enc, max_new_tokens=256, do_sample=False)
    ans = tok.decode(out[0], skip_special_tokens=True)

    # Append simple source footer
    if cites:
        lines = []
        for c in cites:
            bits = []
            if c.get("source"): bits.append(str(c["source"]))
            if c.get("section"): bits.append(f"§ {c['section']}")
            lines.append(" | ".join(bits))
        ans += "\n\nSources:\n- " + "\n- ".join(lines)
    return ans

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What are the learning objectives for Chapter 25?"
    print(answer(q))
