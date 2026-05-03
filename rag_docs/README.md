# rag_docs — TIDE-HF Knowledge Base

Place your PDF documents here. The RAG system indexes them once and stores
embeddings in `rag_db/` (~50 MB). After that it works entirely offline.

## Folder layout

```
rag_docs/
  clinical/     ← shown only to clinicians
  patient/      ← shown only to patients
  shared/       ← shown to both (optional)
```

## Suggested PDFs

### clinical/
| File name | Where to get it |
|---|---|
| `aha_hf_guidelines_2022.pdf` | ahajournals.org (free) |
| `strong_hf_protocol.pdf` | nejm.org/doi/10.1056/NEJMoa2204512 |
| `sacubitril_valsartan_pi.pdf` | drugs.com → Entresto → Full PI |
| `spironolactone_pi.pdf` | drugs.com → Spironolactone → Full PI |
| `carvedilol_pi.pdf` | drugs.com → Carvedilol → Full PI |
| `empagliflozin_pi.pdf` | drugs.com → Jardiance → Full PI |
| `furosemide_pi.pdf` | drugs.com → Furosemide → Full PI |

### patient/
| File name | Where to get it |
|---|---|
| `entresto_patient_guide.pdf` | novartis.com or heart.org |
| `beta_blocker_patient_guide.pdf` | heart.org |
| `chf_lifestyle_guide.pdf` | heart.org |
| `fluid_restriction_guide.pdf` | heart.org |

## Commands

```bash
# Index all PDFs (run once after adding PDFs)
python scripts/setup_rag.py

# Re-index from scratch
python scripts/setup_rag.py --force

# Run a smoke test after indexing
python scripts/setup_rag.py --smoke

# Check how many chunks are indexed
python -c "from chf_titration.rag import TideRAG; r=TideRAG(); print(r.chunk_count, 'chunks')"
```

## Notes

- These folders are in `.gitignore` — PDFs stay local, never uploaded to GitHub
- `rag_db/` is also gitignored — the vector store is rebuilt from PDFs on each machine
- Each PDF is split into ~400-word chunks with 50-word overlap
- Audience tags (clinical / patient) control which chunks each role sees
