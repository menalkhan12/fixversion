import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import chromadb
import numpy as np
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str


def _clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 160) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []
    out: List[str] = []
    i = 0
    while i < len(text):
        out.append(text[i : i + chunk_size])
        if i + chunk_size >= len(text):
            break
        i += max(1, chunk_size - overlap)
    return out


class RagEngine:
    def __init__(
        self,
        data_dir: str,
        persist_dir: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "ist_kb",
    ):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.embedder = SentenceTransformer(embedding_model)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.col = self.client.get_or_create_collection(name=collection_name)

        self.chunks: List[Chunk] = []
        self._bm25 = None
        self._bm25_tokens: List[List[str]] = []

        self._load_and_index()

    def _load_and_index(self) -> None:
        chunks: List[Chunk] = []

        if not os.path.isdir(self.data_dir):
            self.chunks = []
            self._bm25 = BM25Okapi([[]])
            return

        for name in sorted(os.listdir(self.data_dir)):
            path = os.path.join(self.data_dir, name)
            if os.path.isdir(path):
                continue

            if name.lower().endswith(".txt"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
                for idx, piece in enumerate(_chunk_text(raw)):
                    chunks.append(Chunk(chunk_id=f"{name}:{idx}", source=name, text=piece))

            elif name.lower().endswith(".json"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
                try:
                    obj = json.loads(raw)
                except Exception:
                    obj = None

                if isinstance(obj, dict) or isinstance(obj, list):
                    as_text = json.dumps(obj, ensure_ascii=False)
                    for idx, piece in enumerate(_chunk_text(as_text)):
                        chunks.append(Chunk(chunk_id=f"{name}:{idx}", source=name, text=piece))

        self.chunks = chunks

        ids = [c.chunk_id for c in self.chunks]
        existing = set()
        try:
            peek = self.col.get(include=[], limit=1)
            if peek and "ids" in peek and peek["ids"]:
                pass
        except Exception:
            pass

        try:
            all_existing = self.col.get(include=[], limit=100000)
            for _id in all_existing.get("ids", []) or []:
                existing.add(_id)
        except Exception:
            existing = set()

        to_add = [c for c in self.chunks if c.chunk_id not in existing]
        if to_add:
            texts = [c.text for c in to_add]
            embeds = self.embedder.encode(texts, normalize_embeddings=True).tolist()
            self.col.add(
                ids=[c.chunk_id for c in to_add],
                documents=texts,
                metadatas=[{"source": c.source} for c in to_add],
                embeddings=embeds,
            )

        self._bm25_tokens = [_tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(self._bm25_tokens or [[]])

    def _vector_search(self, query: str, k: int) -> Dict[str, float]:
        if not self.chunks:
            return {}
        q = self.embedder.encode([query], normalize_embeddings=True).tolist()[0]
        res = self.col.query(query_embeddings=[q], n_results=min(k, max(1, len(self.chunks))))
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        scores: Dict[str, float] = {}
        for _id, dist in zip(ids, dists):
            try:
                scores[_id] = float(1.0 - dist)
            except Exception:
                scores[_id] = 0.0
        return scores

    def _bm25_search(self, query: str, k: int) -> Dict[str, float]:
        if not self.chunks:
            return {}
        tokens = _tokenize(query)
        scores_arr = self._bm25.get_scores(tokens)
        if scores_arr is None or len(scores_arr) == 0:
            return {}
        idxs = np.argsort(scores_arr)[::-1][: min(k, len(self.chunks))]
        out: Dict[str, float] = {}
        for i in idxs:
            out[self.chunks[int(i)].chunk_id] = float(scores_arr[int(i)])
        return out

    @staticmethod
    def _minmax_norm(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        vals = list(scores.values())
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return {k: 0.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    def hybrid_search(self, query: str, top_k: int = 8) -> List[Dict[str, str]]:
        if not query.strip():
            return []

        vec = self._vector_search(query, k=24)
        kw = self._bm25_search(query, k=24)

        vec_n = self._minmax_norm(vec)
        kw_n = self._minmax_norm(kw)

        combined: Dict[str, float] = {}
        for _id in set(vec_n.keys()) | set(kw_n.keys()):
            combined[_id] = 0.65 * vec_n.get(_id, 0.0) + 0.35 * kw_n.get(_id, 0.0)

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        ranked = ranked[:top_k]

        if ranked and ranked[0][1] >= 0.10:
            return [self._chunk_payload(_id) for _id, _ in ranked]

        fallback = self._vector_search("General IST Admission Overview", k=top_k)
        if not fallback:
            return []
        fb_ranked = sorted(fallback.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [self._chunk_payload(_id) for _id, _ in fb_ranked]

    def _chunk_payload(self, chunk_id: str) -> Dict[str, str]:
        try:
            res = self.col.get(ids=[chunk_id], include=["documents", "metadatas"])
            doc = (res.get("documents") or [[""]])[0][0]
            md = (res.get("metadatas") or [[{}]])[0][0]
            source = md.get("source", "") if isinstance(md, dict) else ""
            return {"id": chunk_id, "source": source, "text": doc}
        except Exception:
            for c in self.chunks:
                if c.chunk_id == chunk_id:
                    return {"id": c.chunk_id, "source": c.source, "text": c.text}
        return {"id": chunk_id, "source": "", "text": ""}
