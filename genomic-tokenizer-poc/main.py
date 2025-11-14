from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from genomic_tokenizer import GenomicTokenizer


app = FastAPI(title="Genomic Motif Tokenizer · POC")

# Allow browser access if you later hit it from another host/port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static UI from /static
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


class TokenizeRequest(BaseModel):
    text: str
    min_motif_len: Optional[int] = 3
    max_motif_len: Optional[int] = 10
    min_freq: Optional[int] = 5
    max_vocab_size: Optional[int] = 2000


class TokenizeResponse(BaseModel):
    tokens: List[int]
    decoded_text: str
    vocab_size: int
    bytes_count: int
    tokens_count: int
    tokens_per_byte: float
    motif_tokens_count: int
    raw_tokens_count: int
    sample_motifs: List[str]


@app.get("/")
def serve_index():
    """
    Serve the Panton × Crawford inspired UI.
    """
    return FileResponse("static/index.html")


@app.post("/api/tokenize", response_model=TokenizeResponse)
def tokenize(req: TokenizeRequest) -> TokenizeResponse:
    """
    Real endpoint: take text, run GenomicTokenizer, return stats + tokens.
    """
    tokenizer = GenomicTokenizer(
        min_motif_len=req.min_motif_len or 3,
        max_motif_len=req.max_motif_len or 10,
        min_freq=req.min_freq or 5,
        max_vocab_size=req.max_vocab_size or 2000,
    )

    corpus = [req.text]
    tokenizer.fit(corpus)

    tokens = tokenizer.encode(req.text)
    decoded = tokenizer.decode(tokens)

    data = req.text.encode("utf-8")
    bytes_count = len(data)
    tokens_count = len(tokens)
    tokens_per_byte = float(tokens_count) / bytes_count if bytes_count else 0.0
    motif_tokens_count = sum(1 for t in tokens if t >= 0)
    raw_tokens_count = tokens_count - motif_tokens_count

    # up to 20 motifs for UI display
    sample_motifs: List[str] = []
    for motif_bytes in list(tokenizer.id_to_motif.values())[:20]:
        try:
            sample_motifs.append(motif_bytes.decode("utf-8", errors="replace"))
        except Exception:
            sample_motifs.append(repr(motif_bytes))

    return TokenizeResponse(
        tokens=tokens,
        decoded_text=decoded,
        vocab_size=len(tokenizer.id_to_motif),
        bytes_count=bytes_count,
        tokens_count=tokens_count,
        tokens_per_byte=tokens_per_byte,
        motif_tokens_count=motif_tokens_count,
        raw_tokens_count=raw_tokens_count,
        sample_motifs=sample_motifs,
    )
