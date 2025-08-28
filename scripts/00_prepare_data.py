import os, json, random, yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split

CFG = yaml.safe_load(open("config.yaml"))
random.seed(CFG.get("seed", 42))
os.makedirs("data", exist_ok=True)

# Load HF dataset (requires internet on first run)
ds = load_dataset("Josephgflowers/Finance-Instruct-500k", split="train")

rows = []
for ex in ds:
    inst = ex.get("user") or ex.get("instruction") or ex.get("question")
    ans  = ex.get("assistant") or ex.get("output") or ex.get("answer")
    if inst and ans:
        rows.append({
            "instruction": str(inst).strip(),
            "answer":      str(ans).strip()
        })

n_train = int(CFG.get("sample_train_size", 3000))
n_val   = int(CFG.get("sample_val_size",   300))
n_test  = int(CFG.get("sample_test_size",  300))

total_needed = n_train + n_val + n_test
if total_needed < len(rows):
    rows = random.sample(rows, total_needed)

rest, test = train_test_split(rows, test_size=n_test, random_state=CFG.get("seed", 42))
train, val = train_test_split(rest, test_size=n_val,  random_state=CFG.get("seed", 42))

def dump(name, data):
    path = f"data/finance_instruct_sample_{name}.jsonl"
    with open(path, "w") as f:
        for r in data:
            obj = {
                "input":  f"instruction: {r['instruction']}\ncontext: ",
                "target": r["answer"]
            }
            f.write(json.dumps(obj) + "\n")
    print(f"Wrote {len(data)} -> {path}")

dump("train", train)
dump("val",   val)
dump("test",  test)
