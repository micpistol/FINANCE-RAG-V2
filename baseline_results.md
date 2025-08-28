# Baseline Results: Base T5-small + RAG (ChromaDB)

**Model:** Base T5-small + RAG (ChromaDB)  
**Eval Set:** 50 sampled questions  
**Date:** 2025-08-27

## Metrics
- **ROUGE-1:** 0.044
- **ROUGE-2:** 0.011
- **ROUGE-L:** 0.035
- **BLEU:** 0.808
- **Mean Cosine Similarity (SBERT):** 0.198

## Notes
- Answers are short/incomplete (expected for untuned base).
- Retrieval is on-topic; citations (source + section) render correctly.
- This snapshot will be our "before" reference for fine-tuning.

*Next: fine-tune T5-small on Finance-Instruct-500k, re-run the same 50Q eval.*
