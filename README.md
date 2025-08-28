# Finance-RAG-v2

A Retrieval-Augmented Generation (RAG) system for finance domain knowledge, built with T5-small, ChromaDB, and sentence transformers.

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd /Users/michael/dev/Finance-RAG-v2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Process Your Documents
```bash
# Convert markdown to chunks
python scripts/01d_convert_markdown_to_chunks.py \
  --input corpus/your-document.md \
  --source-name "your-document.md"

# Build ChromaDB index
python scripts/01c_build_index_chromadb.py
```

### 3. Test RAG System
```bash
# Quick test with base T5 model
python scripts/rag_infer.py "Your question here?"

# Quick evaluation (50 questions)
python scripts/04_eval_fast.py --limit 50
```

## 📁 Project Structure

```
Finance-RAG-v2/
├── config.yaml              # Configuration parameters
├── requirements.txt          # Python dependencies
├── corpus/                  # Put your .txt/.md docs here
├── data/                    # Training/validation/test splits
├── artifacts/               # Models, indices, processed data
└── scripts/
    ├── 00_prepare_data.py   # Create training data from HF dataset
    ├── 01d_convert_markdown_to_chunks.py  # Markdown → chunks
    ├── 01c_build_index_chromadb.py        # Build ChromaDB index
    ├── 02_finetune_t5.py    # Fine-tune T5 on Colab
    ├── rag_infer.py          # RAG inference (main script)
    └── 04_eval_fast.py      # Fast evaluation with limits
```

## 🔧 Configuration

Key parameters in `config.yaml`:
- **Model**: T5-small base model
- **Embedder**: all-MiniLM-L6-v2 for semantic search
- **Training**: 3000 train, 300 val, 300 test samples
- **Chunking**: 350 tokens with 50 token overlap
- **Retrieval**: Top-4 most relevant chunks

## 🎯 Current Performance

### Base T5 + RAG (50 Questions)
- **ROUGE-1**: 0.044
- **ROUGE-2**: 0.011  
- **ROUGE-L**: 0.035
- **BLEU**: 0.808
- **Mean Cosine Similarity**: 0.198

## 🚀 Fine-tuning on Colab

### 1. Push to GitHub
```bash
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

### 2. Colab Setup
```python
# Cell 1: Setup
!nvidia-smi -L
!pip -q install -U pip setuptools wheel
!pip -q install -r https://raw.githubusercontent.com/<USER>/<REPO>/main/requirements.txt

# Cell 2: Get repo
!git clone https://github.com/<USER>/<REPO>.git Finance-RAG-v2
%cd Finance-RAG-v2

# Cell 3: Train
!python scripts/02_finetune_t5.py

# Cell 4: Download model
!zip -r t5-small-finetuned.zip artifacts/t5-small-finetuned -q
from google.colab import files
files.download("t5-small-finetuned.zip")
```

### 3. Local Testing
```bash
# Update config.yaml
finetuned_dir: "/Users/michael/dev/Finance-RAG-v2/artifacts/t5-small-finetuned"

# Test fine-tuned model
python scripts/rag_infer.py "Your question here?"
python scripts/04_eval_fast.py --limit 50
```

## 💡 Key Features

- **ChromaDB Vector Store**: Avoids FAISS segmentation faults on macOS ARM64
- **Section-Aware Chunking**: Preserves document structure and headers
- **Source Citations**: Automatic attribution to document sections
- **Fast Evaluation**: Configurable question limits for quick A/B testing
- **Colab Integration**: GPU-accelerated fine-tuning pipeline

## 🔍 Troubleshooting

### FAISS Issues on macOS
- **Solution**: Use ChromaDB instead (already implemented)
- **Scripts**: `01c_build_index_chromadb.py` and `rag_infer.py`

### Memory Issues on Colab
- **Config**: Reduce `max_source_length` to 512, `train_batch_size` to 4

### Tokenizer Warnings
- **Fix**: Set `export TOKENIZERS_PARALLELISM=false`

## 📊 Evaluation Metrics

- **ROUGE**: Measures answer completeness and relevance
- **BLEU**: Measures answer fluency and accuracy  
- **Cosine Similarity**: Semantic similarity between predicted and reference answers

## 🎯 Next Steps

1. ✅ **Baseline RAG**: Working with base T5 model
2. 🔄 **Fine-tune T5**: Train on Colab with GPU
3. 📊 **Compare Performance**: Base vs. fine-tuned metrics
4. 🚀 **Production Ready**: Deploy fine-tuned model

## 📝 License

MIT License - see LICENSE file for details.
