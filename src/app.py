from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn 
import re
import time
from fastapi import UploadFile, File, HTTPException
import os
import shutil
import numpy as np
from typing import TypedDict, Optional, Dict, Any, List 
from langgraph.graph import StateGraph, START, END
from pypdf import PdfReader
import torch
import transformers
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import chromadb
from rank_bm25 import BM25Okapi

# ----------------------------------------------------------------------------
# CPU thread configuration (do this before heavy libs spin up thread pools)
# ----------------------------------------------------------------------------
NUM_CPU_THREADS = os.cpu_count() or 4
os.environ["OMP_NUM_THREADS"] = str(NUM_CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_CPU_THREADS)
torch.set_num_threads(NUM_CPU_THREADS)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIRECTORY = str(PROJECT_ROOT / "Docs" / "chromadb_dynamic_1")
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# (folder containing PDFs, chroma collection name)
# these docs are just taken as example
COLLECTIONS_CONFIG = [
    (str(PROJECT_ROOT / "data" / "documents" / "ArtificalIntelligence"), "ArtificalIntelligence"),
    (str(PROJECT_ROOT / "data" / "documents" / "CICD"), "CICD"),
    # (str(PROJECT_ROOT / "data" / "documents" / "Deployment"), "MLDeployment"),
    (str(PROJECT_ROOT / "data" / "documents" / "Transformer"), "Transformer"),
]


app = FastAPI(title="RAG API")

class QueryRequest(BaseModel):
    query: str

collections = {}

reranker = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

# Phrases that should make the bot refuse to answer (anti-jailbreak / scope guard).
# NOTE: as written this also blocks legitimate questions containing words like
# "describe", "suggest", "best", "should" -- prune this list for your real use case.
BLOCKED_PHRASES = [
    # Creativity
    "joke", "poem", "song", "riddle", "story", "storytelling", "funny", "rhyme",
    "imagine", "imagination", "creative", "creativity", "fiction", "fairy", "rap", "lyrics",
    "haiku", "verse", "narrate", "narration", "fantasy", "metaphor", "simile", "analogy",
    "dream", "fairytale", "novel", "character", "dialogue", "drama", "scene",
    "plot", "script", "movie", "film", "scenario", "skit",
    # Opinion / Emotion
    "opinion", "believe", "feelings", "emotions", "personal", "your view",
    "your opinion", "philosophy", "if you were", "in your view", "do you agree",
    "do you think", "moral", "ethics", "ethical", "values", "preference", "favorite",
    # Speculative / Hypothetical
    "what if", "predict", "prediction", "forecast", "could happen", "might happen",
    "hypothesis", "suppose", "guess", "hypothetical", "imagine a world", "pretend",
]
BLOCKED_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in BLOCKED_PHRASES) + r")\b", re.IGNORECASE
)


def query_guardrail(query: str) -> bool:
    return bool(BLOCKED_PATTERN.search(query))


# ----------------------------------------------------------------------------
# MODEL LOADING (lazy singletons so the app loads them only when needed)
# ----------------------------------------------------------------------------
embed_model = None
pipe = None


def get_reranker():
    global reranker
    if reranker is None:
        reranker = CrossEncoder(RERANKER_MODEL, activation_fn=torch.nn.Sigmoid())
    return reranker


def get_embed_model() -> SentenceTransformer:
    global embed_model
    if embed_model is None:
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return embed_model


def get_pipe():
    print("Model loading...")
    global pipe
    if pipe is None:
        pipe = transformers.pipeline(
            "text-generation",
            model=MODEL_ID,
            model_kwargs={
                "torch_dtype": DTYPE,
                "low_cpu_mem_usage": True,
            },
            device_map="auto",
            trust_remote_code=True,
        )
        _ = pipe("Hello", max_new_tokens=1, do_sample=False)
    return pipe


def ensure_models_loaded():
    get_embed_model()
    get_pipe()
    get_reranker()

# ----------------------------------------------------------------------------
# DOCUMENT PROCESSING (Table Parsing)
# ----------------------------------------------------------------------------
... (file content omitted for brevity) ...
