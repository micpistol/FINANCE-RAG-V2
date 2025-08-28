import os, re, json, yaml, argparse, pathlib

CFG = yaml.safe_load(open("config.yaml"))
OUT_PATH = CFG.get("corpus_chunks_path", "artifacts/corpus_chunks.jsonl")

HEADER_RE = re.compile(r'^(#{1,6})\s+(.*)$')  # #, ##, ### headers

def read_markdown(path):
    with open(path, "r") as f:
        return f.read()

def md_to_chunks(md_text, source_name, target_words=450, overlap_words=60):
    lines = md_text.splitlines()
    current_section = None
    buffer_words, chunks = [], []

    def flush():
        nonlocal buffer_words, current_section
        if buffer_words:
            text = " ".join(buffer_words).strip()
            if text:
                chunks.append({
                    "text": text,
                    "meta": {
                        "source": source_name,
                        "section": current_section
                    }
                })
            buffer_words = []

    for ln in lines:
        m = HEADER_RE.match(ln.strip())
        if m:
            flush()
            current_section = m.group(2).strip()
            continue

        # normalize bullets and whitespace
        ln = ln.replace("·", "- ").strip()
        ln = re.sub(r'\s+', ' ', ln)
        if ln:
            buffer_words.extend(ln.split())
            if len(buffer_words) >= target_words:
                flush()
                if overlap_words > 0:
                    carry = buffer_words[-overlap_words:]
                    buffer_words = carry.copy()

    flush()
    return chunks

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to markdown-like chapter file")
    ap.add_argument("--source-name", default=None, help="Label stored in meta.source")
    ap.add_argument("--out", default=OUT_PATH, help="Output JSONL path (corpus_chunks.jsonl)")
    args = ap.parse_args()

    text = read_markdown(args.input)
    source_name = args.source_name or pathlib.Path(args.input).name
    chunks = md_to_chunks(text, source_name)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    print(f"Wrote {len(chunks)} chunks -> {args.out}")
