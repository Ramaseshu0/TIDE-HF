"""Launch the TIDE-HF FastAPI backend on port 8000."""
import os

# Disable tokenizer / BLAS / joblib parallelism before any heavy imports.
# sentence-transformers + torch + uvicorn on macOS Python 3.14 otherwise
# fork-segfaults when the RAG engine initializes (loky semaphore leak).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("JOBLIB_START_METHOD", "spawn")

# Ensure 'spawn' multiprocessing instead of fork — fork() on macOS after
# torch is loaded is unsafe.
import multiprocessing as _mp
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import uvicorn

if __name__ == "__main__":
    uvicorn.run("chf_titration.api:app", host="127.0.0.1", port=8000, reload=False)
