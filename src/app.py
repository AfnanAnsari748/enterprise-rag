from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
import re
import time
import asyncio
from contextlib import asynccontextmanager
from fastapi import UploadFile, File, HTTPException
import shutil
import numpy as np
from typing import TypedDict, Optional, Dict, Any, List
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
import torch
import transformers
from pypdf import PdfReader
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
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "Docs/chromadb_dynamic_1"
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

# (folder containing PDFs, chroma collection name)
# these docs are just taken as example
COLLECTIONS_CONFIG = [  
    ("data/documents/ArtificalIntelligence", "ArtificalIntelligence"),
    ("data/documents/CICD", "CICD"),
    # ("data/documents/Deployment", "MLDeployment"),
    ("data/documents/Transformer", "Transformer"),
]


collections_lock = asyncio.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global collections
    print("Loading models and collections...")
    get_embed_model()
    get_pipe()
    collections = load_collections()
    print("API ready.")
    yield

app = FastAPI(title="RAG API", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

collections = {}

reranker = CrossEncoder(RERANKER_MODEL, activation_fn=torch.nn.Sigmoid())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

# Phrases that cause the bot to refuse out-of-scope requests.
# Keep this list narrow — overly broad terms block legitimate technical queries.
BLOCKED_PHRASES = [
    # Clearly creative / entertainment requests
    "joke", "poem", "song", "riddle", "storytelling", "funny", "rhyme",
    "rap", "lyrics", "haiku", "verse", "fairytale", "novel",
    "drama", "movie", "film", "skit",
    # Pure opinion / personal questions directed at the bot
    "your opinion", "your view", "if you were", "in your view",
    "do you agree", "do you think", "do you feel",
    "your favorite", "your preference",
    # Roleplay / jailbreak patterns
    "pretend you are", "act as", "imagine a world", "imagine you are",
]
BLOCKED_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in BLOCKED_PHRASES) + r")\b", re.IGNORECASE
)


def query_guardrail(query: str) -> bool:
    return bool(BLOCKED_PATTERN.search(query))


# ----------------------------------------------------------------------------
# MODEL LOADING (singletons — loaded once in the lifespan handler)
# ----------------------------------------------------------------------------
embed_model = None
pipe = None

def get_embed_model() -> SentenceTransformer:
    global embed_model
    if embed_model is None:
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return embed_model

def get_pipe():
    global pipe
    if pipe is None:
        print("Loading generation model...")
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

# ----------------------------------------------------------------------------
# DOCUMENT PROCESSING (Table Parsing)
# ----------------------------------------------------------------------------
def split_row(line: str) -> list:
    """Tokenise a line at runs of 2+ spaces/tabs into column cells."""
    return [c.strip() for c in re.split(r"[ \t]{2,}", line.strip()) if c.strip()]


def is_table_row(line: str) -> bool: 
    return len(split_row(line)) >= 2


def normalize_table(block: list) -> str:
    """
    Convert a list of table-like lines into a normalized horizontal
    representation.  The first non-empty line is treated as the header;
    subsequent lines are emitted as 'Header: Value' pairs joined by ' | ',
    preserving left-to-right, row-wise reading order.
    """
    rows = [split_row(l) for l in block if l.strip()]
    if not rows:
        return ""
    if len(rows) == 1:
        return " | ".join(rows[0])
    header = rows[0]
    out = [" | ".join(header)]
    for row in rows[1:]:
        width = max(len(header), len(row))
        pairs = []
        for i in range(width):
            h = header[i] if i < len(header) else ""
            v = row[i] if i < len(row) else ""
            if h and v:
                pairs.append(f"{h}: {v}")
            elif v:
                pairs.append(v)
        if pairs:
            out.append(" | ".join(pairs))
    return "\n".join(out)


def find_column_split(lines: list) -> int | None:
    midpoints = []
    for line in lines:
        if len(line) < 50:
            continue
        gaps = list(re.finditer(r" {5,}", line))
        if gaps:
            g = max(gaps, key=lambda m: m.end() - m.start())
            midpoints.append((g.start() + g.end()) // 2)

    if len(midpoints) < max(3, len(lines) // 4):
        return None

    midpoints.sort()
    split = midpoints[len(midpoints) // 2]

    # Both sides must carry content for at least 30 % of lines
    both = sum(
        1 for l in lines
        if l[:split].strip() and len(l) > split and l[split:].strip()
    )
    return split if both >= len(lines) * 0.3 else None


def reorder_double_column(lines: list) -> list:
    split = find_column_split(lines)
    if split is None:
        return lines
    left = [l[:split].rstrip() for l in lines]
    right = [l[split:].lstrip() if len(l) > split else "" for l in lines]
    return [l for l in left if l.strip()] + [r for r in right if r.strip()]


def segment_and_normalize(text: str) -> str:
    lines = reorder_double_column(text.splitlines())
    output = []
    i = 0
    while i < len(lines): 
        if is_table_row(lines[i]):
            block = []
            while i < len(lines):
                if is_table_row(lines[i]):
                    block.append(lines[i])
                    i += 1
                elif (
                    lines[i].strip() == ""
                    and i + 1 < len(lines)
                    and is_table_row(lines[i + 1])
                ):
                    i += 1  # skip blank separator within a table
                else:
                    break
            output.append(normalize_table(block) if len(block) >= 2 else "\n".join(block))
        else:
            output.append(lines[i])
            i += 1
    return "\n".join(output)


def extract_page_text(page) -> str:
    try:
        return page.extract_text(
            extraction_mode="layout",
            layout_mode_space_vertically=False,
        ) or ""
    except TypeError:
        return page.extract_text() or ""


# ----------------------------------------------------------------------------
# DOCUMENT PROCESSING (PDF -> chunks -> embeddings -> chroma)
# ----------------------------------------------------------------------------

def process_pdfs(pdf_path: str, collection_name: str):
    try:

        # --------------------------------------------------
        # If path doesn't exist, create parent directory
        # --------------------------------------------------
        if not os.path.exists(pdf_path):
            parent_dir = os.path.dirname(pdf_path)

            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            print(f"Path not found: {pdf_path}")
            return None

        # --------------------------------------------------
        # If a single PDF was provided
        # --------------------------------------------------
        if os.path.isfile(pdf_path):

            if not pdf_path.lower().endswith(".pdf"):
                print(f"Not a PDF file: {pdf_path}")
                return None

            pdf_files = [(os.path.dirname(pdf_path), os.path.basename(pdf_path))]

        # --------------------------------------------------
        # If a directory was provided
        # --------------------------------------------------
        elif os.path.isdir(pdf_path):

            pdf_files = [
                (pdf_path, f)
                for f in os.listdir(pdf_path)
                if f.lower().endswith(".pdf")
            ]

            if not pdf_files:
                print(f"No PDF files found in {pdf_path}")
                return None

        else:
            print(f"Invalid path: {pdf_path}")
            return None

        print(f"\nFound {len(pdf_files)} PDFs")

        # --------------------------------------------------
        # Chroma
        # --------------------------------------------------
        collection = chroma_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

        existing = collection.get(include=[])
        existing_ids = (
            set(existing["ids"])
            if existing and "ids" in existing
            else set()
        )

        total_chunks = 0

        # --------------------------------------------------
        # Process PDFs
        # --------------------------------------------------
        for directory, pdf_file in pdf_files:
            pdf_full_path = os.path.join(directory,pdf_file)
            print(f"Processing: {pdf_full_path}")

            reader = PdfReader(pdf_full_path)
            text = ""

            for page in reader.pages:
                page_text = extract_page_text(page)
                if page_text:
                    text += (segment_and_normalize(page_text) + "\n")

            print(f"Extracted {len(text)} characters from {pdf_file}")

            if not text.strip():
                print(f"Skipping {pdf_file} (no extractable text)")
                continue

            # --------------------------------------------------
            # Chunking
            # --------------------------------------------------
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150, length_function=len)
            chunks = text_splitter.split_text(text)

            if not chunks:
                print(f"Skipping {pdf_file} (no chunks created)")
                continue

            print(f"Created {len(chunks)} chunks from {pdf_file}")

            # --------------------------------------------------
            # IDs + metadata
            # --------------------------------------------------
            ids = [f"{pdf_file}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "filename": pdf_file,
                    "chunk": i,
                    "category": collection_name
                }
                for i in range(len(chunks))
            ]

            new_chunks = []
            new_ids = []
            new_metadatas = []

            for chunk, cid, meta in zip(
                chunks,
                ids,
                metadatas
            ):

                if cid not in existing_ids:
                    new_chunks.append(chunk)
                    new_ids.append(cid)
                    new_metadatas.append(meta)

            if not new_chunks:
                print( f"Skipping {pdf_file}  (all chunks already stored)")
                continue

            # --------------------------------------------------
            # Embeddings
            # --------------------------------------------------
            embeddings = embed_model.encode(new_chunks)

            if len(embeddings) == 0:
                print(f"Skipping {pdf_file} (no embeddings produced)")
                continue

            # --------------------------------------------------
            # Store in Chroma
            # --------------------------------------------------
            collection.add(
                documents=new_chunks,
                embeddings=embeddings.tolist(),
                ids=new_ids,
                metadatas=new_metadatas
            )

            total_chunks += len(new_chunks)

        print(f"Stored {total_chunks} NEW chunks in {collection_name} collection.")

        return collection

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None


def load_collections():
    results = {}
    for _, collection_name in COLLECTIONS_CONFIG:
        try:
            collection = chroma_client.get_collection(name=collection_name)
            results[collection_name] = collection
            print(f"Loaded collection: {collection_name}")

        except Exception:
            results[collection_name] = None
            print(f"Collection not found: {collection_name}")

    return results

def build_collection_config(uploaded_filename=None):
    config = COLLECTIONS_CONFIG.copy()

    if uploaded_filename:
        upload_path = f"data/documents/uploads/{uploaded_filename}"

        collection_name = uploaded_filename.rsplit(".", 1)[0]
        collection_name = re.sub(
            r"[^a-zA-Z0-9._-]",
            "_",
            collection_name
        )

        config.append((upload_path, collection_name))

    return config


def create_collections(config):
    results = {}

    for dir_path, collection_name in config:

        collection = process_pdfs(
            dir_path,
            collection_name
        )

        if collection is not None:
            results[collection_name] = collection
        else:
            print(
                f"Skipping collection '{collection_name}' "
                f"because ingestion failed."
            )

    return results

def initialize_collections(uploaded_filename=None):
    config = build_collection_config(uploaded_filename)
    return create_collections(config)

# ----------------------------------------------------------------------------
# ROUTING: pick which collection best matches the query
# ----------------------------------------------------------------------------
def route_query(query, collections, threshold=0.35):
    # Create query embedding once
    query_embedding = embed_model.encode(query).tolist()

    best_name = None
    best_score = -1
    best_results = None

    for name, collection in collections.items():
        print("details ", name, collection.metadata)
        if collection is None:
            continue

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents","distances","metadatas"]
            )

        if not results["distances"]:
            continue

        distances = results["distances"][0]
        if not distances:
            continue
        
        # Convert cosine distance → cosine similarity
        similarities = [1 - distance for distance in distances]
        score = max(similarities)
        print(f"{name}: {score:.4f}")

        if score > best_score:
            best_name = name
            best_score = score
            best_results = results

    if best_score >= threshold:
        return ( best_name, best_score, best_results )

    return None, None, None

# ----------------------------------------------------------------------------
# Prompt Building
# ----------------------------------------------------------------------------
def build_chat_prompt(system_prompt: str, user_content: str):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

# ============================================================================
# LANGGRAPH STATE
# ============================================================================
class RAGState(TypedDict, total=False):
    query: str
    domain: Optional[str]
    routing_score: Optional[float]
    collection_data: Optional[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    chunks_evaluation: str
    chunk_evaluation_count: int
    generation_evaluation_count: int
    retrival_failed: str
    context: str
    answer: str
    ans_evaluation: str
    blocked: bool
    retrieval_seconds: float
    generation_seconds: float
    error: Optional[str]

# ============================================================================
# LANGGRAPH NODE 1: QUERY GUARDRAIL
# ============================================================================
def query_guardrail_node(state: RAGState):

    query = state["query"]
    print("\n================ GUARDRAIL ================")

    if query_guardrail(query):
        print("Query blocked")
        return {"blocked": True, "answer": ("I don't know the answer based on the provided context.")}

    print("Query accepted")

    return { "blocked": False }

# ============================================================================
# LANGGRAPH NODE 2: ROUTING
# ============================================================================
def routing_node(state: RAGState):
    query = state["query"]
    print("\n================ ROUTING ================")
    domain_name, score, best_results = route_query(query, collections,threshold=0.35)

    if domain_name is None:
        print("No suitable collection found")
        return {
            "domain": None,
            "routing_score": score,
            "answer": ( "I don't know the answer based on the provided context." )
        }

    print(f"Selected collection: {domain_name} | Score: {score:.4f}")

    return {
        "domain": domain_name,
        "routing_score": score
    }

# ============================================================================
# Candidate RETRIEVAL
# ============================================================================
def hybrid_retrieval(query: str, collection, candidate_k: int = 15, bm25_weight: float = 0.5, semantic_threshold: float = 0.35):

    data = collection.get(include=["documents", "embeddings", "metadatas"])

    all_docs = data["documents"]
    metadata = data["metadatas"]

    all_embeddings = np.array(data["embeddings"], dtype=np.float32)

    if not all_docs:
        return None

    # ============================================================
    # BM25
    # ============================================================
    bm25 = BM25Okapi([doc.split() for doc in all_docs])
    bm25_scores = np.array(bm25.get_scores(query.split()))
    bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max()- bm25_scores.min() + 1e-8)

    # ============================================================
    # Semantic similarity
    # ============================================================
    query_emb = embed_model.encode(query, convert_to_tensor=True)
    semantic_scores = (util.cos_sim(query_emb, all_embeddings)[0].cpu().numpy())
    semantic_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)

    # ============================================================
    # Hybrid score
    # ============================================================
    hybrid_scores = (bm25_weight * bm25_norm + (1 - bm25_weight) * semantic_norm)
    
    print("============================================================")
    print("hybrid_scores:   " , sorted(hybrid_scores))
    print("semantic_scores:   " , semantic_scores)

    valid_indices = np.where(semantic_scores >= semantic_threshold)[0]
    print("valid_indices:   " , valid_indices)
    if len(valid_indices) == 0:
        return None

    # ============================================================
    # Candidate selection
    # ============================================================
    candidate_results = sorted([(i, hybrid_scores[i]) for i in valid_indices], key=lambda x: x[1], reverse=True)[:candidate_k]

    print("candidate_results:  " , candidate_results)

    return {
        "documents": all_docs,
        "metadata": metadata,
        "candidate_results": candidate_results
    }

# ============================================================================
# LANGGRAPH NODE 3: RETRIEVAL
# ============================================================================
def retrieval_node(state: RAGState):

    start_time = time.time()

    query = state["query"]
    domain = state["domain"]

    print("\n================ RETRIEVAL ================")
    print(f"Collection: {domain}")

    if domain is None:
        return {
            "retrieved_chunks": [],
            "retrieval_seconds": 0.0
        }

    collection = collections[domain]

    retrieval_data = hybrid_retrieval(
        query=query,
        collection=collection,
        candidate_k=15,
        bm25_weight=0.5,
        semantic_threshold=0.35
    )

    retrieval_time = time.time() - start_time

    if retrieval_data is None:
        print("No relevant documents found")

        return {
            "collection_data": None,
            "retrieved_chunks": [],
            "retrieval_seconds": retrieval_time
        }

    print(f"Candidates retrieved: {len(retrieval_data['candidate_results'])}")

    return {
        "collection_data": retrieval_data,
        "retrieval_seconds": retrieval_time
    }

# ============================================================================
# RERANKING
# ============================================================================
def rerank_chunks(query: str, retrieval_data: dict, top_k: int = 3):

    all_docs = retrieval_data["documents"]
    metadata = retrieval_data["metadata"]
    candidate_results = retrieval_data["candidate_results"]

    # Candidate documents
    candidate_docs = [all_docs[index] for index, _ in candidate_results]

    # CrossEncoder input
    rerank_pairs = [[query, doc] for doc in candidate_docs]

    # CrossEncoder scores
    rerank_scores = reranker.predict(rerank_pairs)
    print("rerank_scores: ",rerank_scores)

    # Sort by reranker score
    reranked_results = sorted(zip(candidate_results,rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    chunks = []

    for ((index, hybrid_score),rerank_score) in reranked_results:
        chunks.append({
            "text": all_docs[index],
            "metadata": metadata[index],
            "hybrid_score": float(hybrid_score),
            "rerank_score": float(rerank_score)
        })

    return chunks

# ============================================================================
# LANGGRAPH NODE 4: RERANKING
# ============================================================================
def reranking_node(state: RAGState):

    print("\n================ RERANKING ================")

    retrieval_data = state.get("collection_data")
    if retrieval_data is None:
        return {
            "reranked_chunks": []
        }

    query = state["query"]
    chunks = rerank_chunks(query=query, retrieval_data=retrieval_data, top_k=3)
    print(f"Final chunks after reranking: {len(chunks)}")
    print("chunks: ", chunks)

    return {
        "reranked_chunks": chunks
    }

# # ============================================================================
# # EVALUATING CHUNKS
# # ============================================================================
def eval_chunks(chunks):
    
    rerank_scores = sorted([chunk["rerank_score"] for chunk in chunks], reverse=True)
    best_score = rerank_scores[0]
    avg_score = sum(rerank_scores) / len(chunks)

    if best_score > 0.7 and avg_score > 0.5:
        comment = "GOOD"

    else: 
        comment = "BAD" 

    print("Best Chunk Score: " , best_score)
    print("Average Score: " , avg_score)
    print("Comment: ", comment)

    return comment

# # ============================================================================
# # LANGGRAPH NODE 5: RETRIVAL EVALUATOR
# # ============================================================================
def chunk_evaluation_node(state:RAGState):
    print("\n=========== Evaluating Retrival ===========")
    chunks = state.get("reranked_chunks", [])
    chunks_evaluation = eval_chunks(chunks)
    chunk_evaluation_count = state.get("chunk_evaluation_count", 0) + 1

    print("Chunk Evaluation Count:", chunk_evaluation_count)

    return {
        "chunks_evaluation": chunks_evaluation,
        "chunk_evaluation_count": chunk_evaluation_count
    }

# ============================================================================
# LANGGRAPH NODE 6: CONTEXT BUILDING
# ============================================================================
def context_node(state: RAGState):

    print("\n================ CONTEXT ================")
    chunks = state.get("reranked_chunks", [])

    if not chunks:
        return {
            "context": ""
        }

    context = "\n\n".join( chunk["text"] for chunk in chunks)

    # Prevent huge prompt
    context = context[:3000]
    print(f"Context length: {len(context)} characters")

    return {
        "context": context
    }

# ============================================================================
# GENERATION
# ============================================================================
def generate_rag_answer(query: str, domain: str, context: str, max_new_tokens: int = 400):

    system_prompt = (
        f"You are a precise {domain} Q&A bot. "
        f"Answer ONLY using the provided {domain} context. "
        f"If the answer is not in the context, reply exactly: "
        f"'I don't know the answer based on the provided context.' "
        f"Keep your answer short, clear, and under 3 sentences."
    )

    user_content = (
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

    messages = build_chat_prompt(system_prompt, user_content)
    output = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
        repetition_penalty=1.1,
        return_full_text=False
    )

    generated = output[0]["generated_text"]

    if isinstance(generated, list):
        generated = generated[-1]["content"]

    return generated.strip()

# ============================================================================
# LANGGRAPH NODE 7: GENERATION
# ============================================================================
def generation_node(state: RAGState):

    start_time = time.time()
    print("\n================ GENERATION ================")

    query = state["query"]
    domain = state["domain"]
    context = state["context"]

    if not context:
        return {
            "answer": ( "I don't know the answer based on the provided context." ),
            "generation_seconds": 0.0
        }

    answer = generate_rag_answer(
        query=query,
        domain=domain,
        context=context,
        max_new_tokens=400
    )

    generation_time = time.time() - start_time
    print(f"Generation time: {generation_time:.2f}s")

    return {
        "answer": answer,
        "generation_seconds": generation_time
    }


# ============================================================================
# LANGGRAPH NODE 8: GENERATION EVALUATOR
# ============================================================================
def generation_evaluation_node(state: RAGState):

    print("\n================ ANSWER EVALUATOR ================")

    query = state["query"]
    answer = state["answer"]
    generation_evaluation_count = state.get("generation_evaluation_count", 0) + 1

    ans_score = reranker.predict([[query, answer]])

    if ans_score > 0.4:
        ans_evaluation = "GOOD"
    else:
        ans_evaluation = "BAD"

    print("Answer Evaluation Count:", generation_evaluation_count)

    return {
        "ans_evaluation": ans_evaluation,
        "generation_evaluation_count": generation_evaluation_count
    }

# ============================================================================
# CONDITIONAL ROUTING
# ============================================================================
def after_guardrail(state: RAGState):
    if state.get("blocked", False):
        return "blocked"

    return "continue"


def after_routing(state: RAGState):
    if state.get("domain") is None:
        return "no_domain"

    return "continue"

def after_evaluation(state: RAGState):
    chunk_evaluation_count = state.get("chunk_evaluation_count", 0)
    print("chunk_evaluation_count:", chunk_evaluation_count)

    if state.get("chunks_evaluation") == "GOOD":
        return "continue"

    if chunk_evaluation_count < 3:
        return "BAD"

    return "Failed"

def after_generation(state: RAGState):
    generation_evaluation_count = state.get("generation_evaluation_count", 0)

    if state.get("ans_evaluation") == "GOOD":
        return "continue"

    if generation_evaluation_count < 3:
        return "BAD"

    return "Failed"

# ============================================================================
# BUILD RAG GRAPH
# ============================================================================
def build_rag_graph():

    graph = StateGraph(RAGState)

    # ------------------------------------------------------------
    # Add nodes
    # ------------------------------------------------------------
    graph.add_node("query_guardrail", query_guardrail_node)
    graph.add_node("routing", routing_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("reranking", reranking_node)
    graph.add_node("chunk_evaluator", chunk_evaluation_node)
    graph.add_node("context", context_node)
    graph.add_node("generation", generation_node)
    graph.add_node("generation_evaluation", generation_evaluation_node)


    # ------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------
    graph.add_edge(START, "query_guardrail")
    graph.add_conditional_edges("query_guardrail", after_guardrail,{"continue": "routing","blocked": END})
    graph.add_conditional_edges("routing",after_routing,{"continue": "retrieval", "no_domain": END})
    graph.add_edge("retrieval", "reranking")
    graph.add_edge("reranking", "chunk_evaluator")
    graph.add_conditional_edges("chunk_evaluator",after_evaluation,{"continue": "context", "BAD": "routing", "Failed": END})
    graph.add_edge("context", "generation")
    graph.add_edge("generation", "generation_evaluation")
    graph.add_conditional_edges("generation_evaluation",after_generation,{"continue": END, "BAD": "generation", "Failed": END})

    app = graph.compile()

    png_bytes = app.get_graph().draw_mermaid_png()
    with open("rag_graph.png", "wb") as f:
        f.write(png_bytes)

    return app

rag_graph = build_rag_graph()

def get_rag_answer(query: str):

    initial_state = {"query": query, "chunk_evaluation_count": 0, "generation_evaluation_count": 0}
    result = rag_graph.invoke(initial_state)

    return {
        "answer": result.get("answer","I don't know the answer based on the provided context."),
        "domain": result.get("domain"),
        "retrieval_seconds": result.get("retrieval_seconds",0.0),
        "generation_seconds": result.get("generation_seconds",0.0),
        "chunks": result.get("reranked_chunks",[]),
        "blocked": result.get("blocked",False)
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/create_embeddings")
async def create_embeddings():
    global collections

    try:
        async with collections_lock:
            collections = initialize_collections()

        return {
            "status": "success",
            "message": "Embeddings created successfully"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build embeddings: {str(e)}"
        )

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    global collections

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    upload_dir = "data/documents/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    safe_filename = os.path.basename(file.filename)
    file_path = os.path.join(upload_dir, safe_filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"Uploaded file: {safe_filename}")

        async with collections_lock:
            collections = initialize_collections(safe_filename)

        return {
            "status": "success",
            "message": "File uploaded and ingested successfully",
            "filename": safe_filename
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest file: {str(e)}"
        )


@app.post("/query")
async def query(request: QueryRequest):
    return get_rag_answer(query=request.query)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)