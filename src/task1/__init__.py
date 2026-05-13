"""src.task1 - Task 1 sentiment regression (VADER, TF-IDF+Ridge, DeBERTa+LoRA).

Modules:
    splits          70/10/20 stratified split, shared across all 3 models
    vader           rule-based baseline (skipped if vaderSentiment missing)
    tfidf_ridge     TF-IDF + Ridge linear baseline
    deberta         DeBERTa-v3 + LoRA (resume-aware training, chunked scoring)
    deberta_internals  private helpers, callbacks, cache logic
    compare         daily TS comparison + 8 plots
    summary         final output manifest + data/ hygiene check
"""
