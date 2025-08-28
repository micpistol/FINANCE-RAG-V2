import os, json, yaml, subprocess, sys, argparse
import evaluate
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

CFG = yaml.safe_load(open("config.yaml"))

# Metrics
rouge = evaluate.load("rouge")
bleu  = evaluate.load("sacrebleu")
embedder = SentenceTransformer(CFG.get("embedder_name", "sentence-transformers/all-MiniLM-L6-v2"))

TEST_PATH = "data/finance_instruct_sample_test.jsonl"
INFER_SCRIPT = os.path.join("scripts", "rag_infer.py")

def gen_answer(question: str) -> str:
    # Call rag_infer.py as a subprocess to avoid tight coupling
    out = subprocess.check_output([sys.executable, INFER_SCRIPT, question])
    return out.decode().strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to evaluate")
    args = parser.parse_args()
    
    assert os.path.exists(TEST_PATH), f"Missing test set: {TEST_PATH}. Run scripts/00_prepare_data.py first."
    assert os.path.exists(INFER_SCRIPT), f"Missing {INFER_SCRIPT}"

    refs, hyps, sims = [], [], []
    
    # Read all test data first
    with open(TEST_PATH) as f:
        test_data = [json.loads(line) for line in f]
    
    # Apply limit if specified
    if args.limit:
        test_data = test_data[:args.limit]
        print(f"Evaluating on {len(test_data)} questions (limited from full test set)")
    else:
        print(f"Evaluating on all {len(test_data)} questions")

    for ex in tqdm(test_data, desc="Evaluating"):
        # Our jsonl 'input' is "instruction: ...\ncontext: " — extract the instruction part
        q = ex["input"].split("instruction:", 1)[-1].split("context:", 1)[0].strip()
        ref = ex["target"].strip()

        hyp = gen_answer(q)
        refs.append(ref)
        hyps.append(hyp)

        # Cosine similarity as a loose relevance/fluency proxy
        e_ref = embedder.encode(ref, convert_to_tensor=True, normalize_embeddings=True)
        e_hyp = embedder.encode(hyp, convert_to_tensor=True, normalize_embeddings=True)
        sims.append(util.cos_sim(e_ref, e_hyp).item())

    rouge_res = rouge.compute(predictions=hyps, references=refs)
    bleu_res  = bleu.compute(predictions=hyps, references=[[r] for r in refs])
    mean_sim  = sum(sims) / max(1, len(sims))

    print(f"\n=== Evaluation Results ({len(test_data)} questions) ===")
    print("ROUGE:", rouge_res)
    print("BLEU:", bleu_res)
    print(f"Mean cosine similarity (SBERT): {mean_sim:.4f}")

if __name__ == "__main__":
    main()
