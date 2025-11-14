import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

from genomic_tokenizer import GenomicTokenizer


class GenomicHandler(SimpleHTTPRequestHandler):
    """
    Simple HTTP server:

    - GET /            -> serves static/index.html
    - GET /static/...  -> serves files from ./static
    - POST /api/tokenize -> runs GenomicTokenizer and returns JSON
    """

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "/static/index.html"
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/tokenize":
            self.send_error(404, "Not Found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "Invalid Content-Length")
            return

        body = self.rfile.read(length)
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            self.send_error(400, "Invalid JSON")
            return

        try:
            response = self.handle_tokenize(data)
        except Exception as exc:  # last-resort guard
            self.send_error(500, f"Internal error: {exc}")
            return

        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def handle_tokenize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        text = str(data.get("text", ""))

        min_motif_len = int(data.get("min_motif_len", 3) or 3)
        max_motif_len = int(data.get("max_motif_len", 10) or 10)
        min_freq = int(data.get("min_freq", 5) or 5)
        max_vocab_size_raw = data.get("max_vocab_size", 2000)
        max_vocab_size = int(max_vocab_size_raw) if max_vocab_size_raw is not None else None

        tokenizer = GenomicTokenizer(
            min_motif_len=min_motif_len,
            max_motif_len=max_motif_len,
            min_freq=min_freq,
            max_vocab_size=max_vocab_size,
        )

        # For now, fit on this single text (POC)
        corpus: List[str] = [text]
        tokenizer.fit(corpus)

        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        data_bytes = text.encode("utf-8")
        bytes_count = len(data_bytes)
        tokens_count = len(tokens)
        tokens_per_byte = float(tokens_count) / bytes_count if bytes_count else 0.0

        motif_tokens_count = sum(1 for t in tokens if t >= 0)
        raw_tokens_count = tokens_count - motif_tokens_count

        sample_motifs: List[str] = []
        for motif_bytes in list(tokenizer.id_to_motif.values())[:20]:
            try:
                sample_motifs.append(motif_bytes.decode("utf-8", errors="replace"))
            except Exception:
                sample_motifs.append(repr(motif_bytes))

        return {
            "tokens": tokens,
            "decoded_text": decoded,
            "vocab_size": len(tokenizer.id_to_motif),
            "bytes_count": bytes_count,
            "tokens_count": tokens_count,
            "tokens_per_byte": tokens_per_byte,
            "motif_tokens_count": motif_tokens_count,
            "raw_tokens_count": raw_tokens_count,
            "sample_motifs": sample_motifs,
        }


def run_server(port: int = 8000) -> None:
    # Serve files relative to this script (so ./static works)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, GenomicHandler)
    print(f"Serving at http://localhost:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
