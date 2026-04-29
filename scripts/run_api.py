"""Launch the TIDE-HF FastAPI backend on port 8000."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("chf_titration.api:app", host="127.0.0.1", port=8000, reload=False)
