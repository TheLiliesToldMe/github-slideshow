"""Motif-based genomic tokenizer utilities."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional


class GenomicTokenizer:
    """
    Very simple genomic-style motif tokenizer on bytes.

    - fit(corpus_texts): learns a motif dictionary from raw texts.
    - encode(text): returns a list of integer token IDs.
    - decode(token_ids): reconstructs the original text.

    Motif IDs >= 0
    Raw bytes are encoded as negative IDs: raw_id = raw_byte_offset - byte_value
    """

    def __init__(
        self,
        min_motif_len: int = 3,
        max_motif_len: int = 12,
        min_freq: int = 10,
        max_vocab_size: Optional[int] = 5000,
    ):
        self.min_motif_len = min_motif_len
        self.max_motif_len = max_motif_len
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

        # id -> motif bytes
        self.id_to_motif: Dict[int, bytes] = {}
        # motif bytes -> id
        self.motif_to_id: Dict[bytes, int] = {}

        # reserve <RAW_BYTE> offset space for raw bytes
        # negative IDs will represent raw bytes: -1 = 0x00, -2 = 0x01, ...
        self.raw_byte_offset = -1

    def fit(self, texts: List[str], encoding: str = "utf-8") -> None:
        """Learn motif dictionary from a list of texts."""

        motif_counts = Counter()

        for text in texts:
            data = text.encode(encoding)
            length = len(data)
            for i in range(length):
                # collect n-grams from min_len to max_len
                for motif_len in range(self.min_motif_len, self.max_motif_len + 1):
                    j = i + motif_len
                    if j > length:
                        break
                    motif = data[i:j]
                    motif_counts[motif] += 1

        # filter by frequency
        motifs = [
            (motif, freq)
            for motif, freq in motif_counts.items()
            if len(motif) >= self.min_motif_len and freq >= self.min_freq
        ]

        # sort motifs by approximate "compression gain": freq * (len-1)
        motifs.sort(key=lambda item: item[1] * (len(item[0]) - 1), reverse=True)

        if self.max_vocab_size is not None and len(motifs) > self.max_vocab_size:
            motifs = motifs[: self.max_vocab_size]

        # assign IDs starting from 0
        self.id_to_motif = {}
        self.motif_to_id = {}
        for idx, (motif, _freq) in enumerate(motifs):
            self.id_to_motif[idx] = motif
            self.motif_to_id[motif] = idx

    def encode(self, text: str, encoding: str = "utf-8") -> List[int]:
        """Encode a single text into token IDs."""

        data = text.encode(encoding)
        length = len(data)
        tokens: List[int] = []
        index = 0

        # If there are no motifs, emit raw-byte tokens only.
        if not self.id_to_motif:
            for byte_val in data:
                tokens.append(self.raw_byte_offset - byte_val)
            return tokens

        # Precompute motifs by length for faster lookup
        motifs_by_len: Dict[int, Dict[bytes, int]] = {}
        for motif, idx in self.motif_to_id.items():
            motif_length = len(motif)
            motifs_by_len.setdefault(motif_length, {})[motif] = idx

        max_length = max(motifs_by_len)
        min_length = min(motifs_by_len)

        while index < length:
            matched = False
            # Try longest motifs first (greedy maximal match)
            for motif_length in range(max_length, min_length - 1, -1):
                end = index + motif_length
                if end > length:
                    continue
                fragment = data[index:end]
                maybe = motifs_by_len.get(motif_length)
                if maybe is None:
                    continue
                motif_id = maybe.get(fragment)
                if motif_id is not None:
                    tokens.append(motif_id)
                    index = end
                    matched = True
                    break

            if not matched:
                # emit raw byte as negative ID
                byte_val = data[index]
                tokens.append(self.raw_byte_offset - byte_val)
                index += 1

        return tokens

    def decode(self, token_ids: List[int], encoding: str = "utf-8") -> str:
        """Decode token IDs back to text."""

        chunks: List[bytes] = []

        for token_id in token_ids:
            if token_id >= 0:
                motif = self.id_to_motif.get(token_id)
                if motif is None:
                    raise ValueError(f"Unknown motif ID: {token_id}")
                chunks.append(motif)
            else:
                # raw byte: reverse mapping
                byte_val = (self.raw_byte_offset - token_id) & 0xFF
                chunks.append(bytes([byte_val]))

        data = b"".join(chunks)
        return data.decode(encoding, errors="replace")
