# Enterprise Chatbot (RAG)

An enterprise-focused Retrieval-Augmented Generation (RAG) reference implementation. This repository provides a production-oriented skeleton for building a domain-specific Q&A chatbot that
- stores document embeddings in ChromaDB
- retrieves and re-ranks candidates using a hybrid BM25 + semantic retrieval
- generates evidence-backed answers with a local LLM

This repo was structured and packaged for clarity and maintainability. The focus is on reproducible ingestion and secure serving through an API.

Key features
- FastAPI-based HTTP API with endpoints to ingest PDFs and query the RAG system
- Document processing pipeline: PDF -> text extraction -> table normalization -> text chunking -> embeddings -> ChromaDB
- Hybrid retrieval (BM25 + semantic) plus CrossEncoder reranking
- LangGraph workflow to orchestrate guardrails, routing, retrieval, reranking, generation, and evaluation

Repository layout
- src/ - application source (FastAPI app)
- data/ - store your PDFs in data/documents/ and uploaded files get written to data/documents/uploads/
- Docs/ - ChromaDB persistence (excluded from the repo to keep the repository small)
- docs/ - documentation and images
- tests/ - unit and integration tests (empty in this initial import)
- scripts/ - helper scripts for ingestion and maintenance

Quickstart (local)
1. Create and activate a virtual environment:
   python -m venv .venv
   .\\.venv\\Scripts\\activate  # Windows

2. Install dependencies:
   pip install -r requirements.txt

3. Start the API from the project root:
   python -m uvicorn src.app:app --host 0.0.0.0 --port 8000

4. Healthcheck:
   GET /health  -> {"status":"ok"}

5. Rebuild embeddings (careful: heavy):
   POST /create_embeddings

6. Upload and ingest a PDF:
   POST /ingest (multipart form with file field)

7. Query the RAG system:
   POST /query  {"query":"your question"}

Configuration
- Update MODEL_ID, RERANKER_MODEL, EMBED_MODEL_NAME and other constants in src/app.py to match the models you want to use.
- The project is designed to load heavy models lazily; ensure the host has adequate CPU/GPU and memory.

Skills & Technologies
- Python, FastAPI, asynchronous APIs
- ChromaDB embeddings, vector similarity search
- SentenceTransformers embeddings & CrossEncoder reranker
- BM25 ranking (rank_bm25)
- LangGraph for orchestration
- PDF parsing and robust table extraction

Contributing
- Open issues and PRs; please follow conventional commits for clarity.
- Large binary files (model weights, chroma persistence, PDFs) should be stored outside the git repo or in a releases storage.

License
- MIT (see LICENSE)

Contact
- Maintainer: (Add your name/email here)
