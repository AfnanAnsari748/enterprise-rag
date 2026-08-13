from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
import re
import time
from fastapi import UploadFile, File, HTTPException
import os
import shutil
import numpy as np

import torch
import transformers
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import chromadb
from rank_bm25 import BM25Okapi

# (file content omitted for brevity)
