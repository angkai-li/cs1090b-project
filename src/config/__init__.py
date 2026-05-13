"""src.config - environment, paths, hyperparameters, runtime state.

Typical notebook usage:
    # First cell (BEFORE any transformers/HF import):
    from src.config.env import setup_environment
    setup_environment()

    # After install/verify done:
    from src.config import paths, hyperparams, runtime
    from src.config.paths import PROJECT_DIR, CACHE_DIR, T1_DEBERTA_MODEL_DIR
    from src.config.hyperparams import MIN_REVIEWS_TASK2, EMBED_DIM, RANDOM_STATE
    from src.config.runtime import device, USE_CUDA, HAS_TRANSFORMERS
"""
