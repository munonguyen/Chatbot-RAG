from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[0-9a-zA-ZÀ-ỹ_]+", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
DEFAULT_ALLOWED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".log"}
MAX_KNOWLEDGE_FILE_BYTES = 2_000_000
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "va",
    "và",
    "la",
    "là",
    "cua",
    "của",
    "co",
    "có",
    "cho",
    "trong",
    "khi",
    "thi",
    "thì",
    "neu",
    "nếu",
    "mot",
    "một",
    "cac",
    "các",
    "nhung",
    "những",
    "duoc",
    "được",
    "toi",
    "tôi",
    "ban",
    "bạn",
    "nay",
    "này",
    "kia",
    "voi",
    "với",
    "hay",
    "hãy",
    "can",
    "cần",
    "the",
    "thế",
    "nao",
    "nào",
}

DOMAIN_KEYWORDS = {
    "GenAI/RAG": {
        "ai",
        "rag",
        "llm",
        "chatbot",
        "embedding",
        "vector",
        "retrieval",
        "prompt",
        "semantic",
        "chunk",
        "index",
        "model",
        "genai",
    },
    "Phần mềm": {
        "api",
        "backend",
        "frontend",
        "server",
        "database",
        "code",
        "python",
        "javascript",
        "deploy",
        "bug",
        "test",
    },
    "Tài chính": {
        "tai",
        "chinh",
        "finance",
        "ke",
        "toan",
        "doanh",
        "thu",
        "loi",
        "nhuan",
        "von",
        "dau",
        "tu",
        "ngan",
        "hang",
    },
    "Y tế": {
        "y",
        "te",
        "benh",
        "vien",
        "thuoc",
        "bac",
        "si",
        "chan",
        "doan",
        "dieu",
        "tri",
        "suc",
        "khoe",
    },
    "Pháp lý": {
        "phap",
        "ly",
        "hop",
        "dong",
        "luat",
        "dieu",
        "khoan",
        "quy",
        "dinh",
        "tranh",
        "chap",
        "tu",
        "van",
    },
    "Giáo dục": {
        "giao",
        "duc",
        "hoc",
        "vien",
        "giang",
        "day",
        "chuong",
        "trinh",
        "bai",
        "tap",
        "ky",
        "nang",
    },
    "Marketing": {
        "marketing",
        "thuong",
        "hieu",
        "khach",
        "hang",
        "chien",
        "dich",
        "content",
        "seo",
        "ads",
        "ban",
        "hang",
    },
    "Chăm sóc khách hàng": {
        "khach",
        "hang",
        "ho",
        "tro",
        "ticket",
        "enterprise",
        "business",
        "hoan",
        "sla",
        "hotline",
        "support",
    },
}


@dataclass
class DocumentRecord:
    id: str
    name: str
    content_hash: str
    created_at: float
    chunk_count: int
    source_type: str = "manual"
    source_path: str = ""


@dataclass
class ChunkRecord:
    id: str
    document_id: str
    document_name: str
    index: int
    text: str
    embedding: list[float]


@dataclass
class SearchHit:
    chunk_id: str
    document_id: str
    document_name: str
    index: int
    text: str
    score: float
    source_type: str = "manual"


class ChatMemory:
    def __init__(self, path: Path, max_turns: int = 80) -> None:
        self.path = path
        self.max_turns = max_turns
        self.turns: list[dict[str, Any]] = []
        self.profile_terms: Counter[str] = Counter()
        self.load()

    def observe(self, question: str, answer: str, hits: list[SearchHit]) -> None:
        relevant_hits = [hit for hit in hits if hit.score > 0.03] or hits[:1]
        source_text = " ".join(f"{hit.document_name} {hit.text[:600]}" for hit in relevant_hits[:4])
        question_terms = self._profile_terms(question)
        source_terms = self._profile_terms(source_text)
        self.profile_terms.update(question_terms)
        self.profile_terms.update(question_terms)
        self.profile_terms.update(source_terms)
        self.turns.append(
            {
                "id": stable_id(str(time.time()), question),
                "created_at": time.time(),
                "question": question,
                "answer_excerpt": answer[:900],
                "sources": [
                    {
                        "document_name": hit.document_name,
                        "source_type": hit.source_type,
                        "score": hit.score,
                    }
                    for hit in relevant_hits[:4]
                ],
            }
        )
        self.turns = self.turns[-self.max_turns :]
        self.save()

    def enhanced_query(self, question: str) -> str:
        profile = self.profile()
        recent_questions = " ".join(turn["question"] for turn in self.turns[-5:])
        terms = " ".join(profile["top_terms"][:12])
        return clean_text(f"{question}\n{profile['domain']}\n{terms}\n{recent_questions}")

    def profile(self) -> dict[str, Any]:
        top_terms = [term for term, _ in self.profile_terms.most_common(16)]
        return {
            "domain": self.infer_domain(top_terms),
            "top_terms": top_terms,
            "turn_count": len(self.turns),
        }

    def prompt_context(self) -> str:
        profile = self.profile()
        recent = self.turns[-6:]
        lines = [
            f"Chuyên ngành suy luận: {profile['domain']}",
            f"Từ khóa người dùng/source nổi bật: {', '.join(profile['top_terms'][:12]) or 'chưa đủ dữ liệu'}",
            "Lịch sử gần đây:",
        ]
        for turn in recent:
            lines.append(f"- Q: {turn['question']}")
            if turn.get("answer_excerpt"):
                lines.append(f"  A: {turn['answer_excerpt'][:220]}")
        return "\n".join(lines)

    def build_research_note(self, latest_question: str, hits: list[SearchHit]) -> str:
        profile = self.profile()
        lines = [
            "# Auto research memory",
            "",
            f"Chuyên ngành suy luận: {profile['domain']}",
            f"Từ khóa trọng tâm: {', '.join(profile['top_terms'][:16]) or 'chưa đủ dữ liệu'}",
            f"Câu hỏi mới nhất: {latest_question}",
            "",
            "## Lịch sử hội thoại gần đây",
        ]
        for turn in self.turns[-8:]:
            lines.append(f"- Người dùng hỏi: {turn['question']}")
            if turn.get("answer_excerpt"):
                lines.append(f"  Trả lời trước đó: {turn['answer_excerpt'][:260]}")

        lines.extend(["", "## Nguồn truy xuất đáng chú ý"])
        for index, hit in enumerate(hits[:6], start=1):
            excerpt = clean_text(hit.text[:900])
            lines.append(
                f"- [R{index}] {hit.document_name} ({hit.source_type}, score {hit.score}): {excerpt}"
            )
        lines.extend(
            [
                "",
                "## Hướng nghiên cứu tiếp theo",
                "- Ưu tiên nguồn có cùng chuyên ngành với hồ sơ người dùng.",
                "- Khi câu hỏi tiếp theo thiếu ngữ cảnh, dùng lịch sử hội thoại và từ khóa trọng tâm để mở rộng truy vấn.",
                "- Nếu nguồn truy xuất không đủ, trả lời rõ là chưa đủ dữ liệu thay vì suy đoán.",
            ]
        )
        return "\n".join(lines)

    def clear(self) -> None:
        self.turns = []
        self.profile_terms = Counter()
        self.save()

    def status(self) -> dict[str, Any]:
        return self.profile()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "turns": self.turns,
            "profile_terms": dict(self.profile_terms),
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.turns = list(payload.get("turns", []))[-self.max_turns :]
        self.profile_terms = Counter(payload.get("profile_terms", {}))

    @staticmethod
    def infer_domain(terms: list[str]) -> str:
        if not terms:
            return "Đang học từ hội thoại"
        term_set = set(terms)
        scores = {
            domain: len(term_set & keywords)
            for domain, keywords in DOMAIN_KEYWORDS.items()
        }
        best_domain, best_score = max(scores.items(), key=lambda item: item[1])
        if best_score <= 0:
            return f"Chủ đề: {', '.join(terms[:3])}"
        return best_domain

    @staticmethod
    def _profile_terms(text: str) -> list[str]:
        return [
            token
            for token in tokenize(text)
            if len(token) >= 3 and not token.isdigit() and token not in STOP_WORDS
        ]


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_diacritics(text: str) -> str:
    text = text.lower().replace("đ", "d").replace("Đ", "d")
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def tokenize(text: str) -> list[str]:
    normalized = strip_diacritics(text)
    tokens = [token for token in TOKEN_RE.findall(normalized) if len(token) > 1]
    return [token for token in tokens if token not in STOP_WORDS]


def stable_id(*parts: str, size: int = 16) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:size]


class TextChunker:
    def __init__(self, max_words: int = 170, overlap_words: int = 35) -> None:
        self.max_words = max_words
        self.overlap_words = overlap_words

    def split(self, text: str) -> list[str]:
        cleaned = clean_text(text)
        if not cleaned:
            return []
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
        chunks: list[str] = []
        current: list[str] = []

        for paragraph in paragraphs:
            words = paragraph.split()
            if len(words) > self.max_words:
                self._flush(current, chunks)
                chunks.extend(self._split_long_words(words))
                current = []
                continue

            if len(current) + len(words) > self.max_words:
                self._flush(current, chunks)
                current = current[-self.overlap_words :] if self.overlap_words else []
            current.extend(words)

        self._flush(current, chunks)
        return chunks

    def _split_long_words(self, words: list[str]) -> list[str]:
        chunks = []
        step = max(1, self.max_words - self.overlap_words)
        for start in range(0, len(words), step):
            window = words[start : start + self.max_words]
            if window:
                chunks.append(" ".join(window))
        return chunks

    @staticmethod
    def _flush(words: list[str], chunks: list[str]) -> None:
        if words:
            chunks.append(" ".join(words).strip())


class HashingEmbedder:
    """Small deterministic embedder for a dependency-free local RAG prototype."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.name = f"local-hashing-{dimensions}"

    def embed(self, text: str) -> list[float]:
        features = self._features(text)
        vector = [0.0] * self.dimensions
        for feature, weight in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big")
            index = raw % self.dimensions
            sign = 1.0 if (raw >> 8) & 1 else -1.0
            vector[index] += sign * weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _features(self, text: str) -> list[tuple[str, float]]:
        tokens = tokenize(text)
        features: list[tuple[str, float]] = [(token, 1.0) for token in tokens]
        features.extend((f"{left}_{right}", 1.35) for left, right in zip(tokens, tokens[1:]))
        for token in tokens:
            if len(token) >= 5:
                features.extend((f"char:{token[i:i + 4]}", 0.35) for i in range(len(token) - 3))
        return features


class VectorStore:
    def __init__(self, path: Path, embedder: HashingEmbedder | None = None) -> None:
        self.path = path
        self.embedder = embedder or HashingEmbedder()
        self.documents: dict[str, DocumentRecord] = {}
        self.chunks: list[ChunkRecord] = []
        self.load()

    def add_document(
        self,
        name: str,
        content: str,
        chunker: TextChunker,
        source_type: str = "manual",
        source_path: str = "",
    ) -> dict[str, Any]:
        cleaned = clean_text(content)
        if not cleaned:
            raise ValueError(f"{name} has no readable text")

        content_hash = stable_id(cleaned, size=32)
        document_id = stable_id(source_type, source_path or name, content_hash)
        self.remove_documents_by_identity(name, source_type, source_path)

        raw_chunks = chunker.split(cleaned)
        chunk_records = []
        for index, chunk_text in enumerate(raw_chunks):
            chunk_id = stable_id(document_id, str(index), chunk_text)
            chunk_records.append(
                ChunkRecord(
                    id=chunk_id,
                    document_id=document_id,
                    document_name=name,
                    index=index,
                    text=chunk_text,
                    embedding=self.embedder.embed(chunk_text),
                )
            )

        self.documents[document_id] = DocumentRecord(
            id=document_id,
            name=name,
            content_hash=content_hash,
            created_at=time.time(),
            chunk_count=len(chunk_records),
            source_type=source_type,
            source_path=source_path,
        )
        self.chunks.extend(chunk_records)
        self.save()
        return {
            "document_id": document_id,
            "name": name,
            "chunk_count": len(chunk_records),
            "content_hash": content_hash,
            "source_type": source_type,
            "source_path": source_path,
        }

    def remove_document(self, document_id: str) -> None:
        self.documents.pop(document_id, None)
        self.chunks = [chunk for chunk in self.chunks if chunk.document_id != document_id]

    def remove_documents_by_identity(self, name: str, source_type: str, source_path: str) -> None:
        matching_ids = []
        for document_id, document in self.documents.items():
            if document.source_type != source_type:
                continue
            if source_path and document.source_path == source_path:
                matching_ids.append(document_id)
            elif not source_path and document.name == name:
                matching_ids.append(document_id)
        for document_id in matching_ids:
            self.remove_document(document_id)

    def clear(self) -> dict[str, Any]:
        self.documents.clear()
        self.chunks.clear()
        self.save()
        return {"cleared": True, "status": self.status()}

    def search(self, query: str, top_k: int = 4) -> list[SearchHit]:
        if not self.chunks:
            return []
        query_embedding = self.embedder.embed(query)
        query_tokens = set(tokenize(query))
        hits = []
        for chunk in self.chunks:
            semantic = dot(query_embedding, chunk.embedding)
            lexical = self._lexical_bonus(query_tokens, chunk.text)
            score = semantic + lexical
            hits.append(
                SearchHit(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    index=chunk.index,
                    text=chunk.text,
                    score=round(score, 4),
                    source_type=self.documents.get(chunk.document_id, DocumentRecord(
                        id=chunk.document_id,
                        name=chunk.document_name,
                        content_hash="",
                        created_at=0,
                        chunk_count=0,
                    )).source_type,
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[: max(1, min(top_k, 10))]

    def status(self) -> dict[str, Any]:
        docs = sorted(self.documents.values(), key=lambda doc: doc.created_at, reverse=True)
        return {
            "document_count": len(self.documents),
            "chunk_count": len(self.chunks),
            "embedding_mode": self.embedder.name,
            "documents": [
                {
                    "id": doc.id,
                    "name": doc.name,
                    "chunk_count": doc.chunk_count,
                    "created_at": doc.created_at,
                    "source_type": doc.source_type,
                    "source_path": doc.source_path,
                }
                for doc in docs
            ],
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": [asdict(doc) for doc in self.documents.values()],
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.documents = {
            item["id"]: DocumentRecord(
                id=item["id"],
                name=item["name"],
                content_hash=item["content_hash"],
                created_at=item["created_at"],
                chunk_count=item["chunk_count"],
                source_type=item.get("source_type", "manual"),
                source_path=item.get("source_path", ""),
            )
            for item in payload.get("documents", [])
        }
        self.chunks = [ChunkRecord(**item) for item in payload.get("chunks", [])]

    @staticmethod
    def _lexical_bonus(query_tokens: set[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        chunk_tokens = set(tokenize(text))
        overlap = len(query_tokens & chunk_tokens)
        return min(0.25, overlap / max(8, len(query_tokens)) * 0.18)


class RAGService:
    def __init__(self, store_path: Path, knowledge_dir: Path | None = None) -> None:
        self.chunker = TextChunker()
        self.store = VectorStore(store_path)
        self.memory = ChatMemory(store_path.parent / "chat_memory.json")
        self.knowledge_dir = knowledge_dir or store_path.parent.parent / "knowledge"
        if os.getenv("AUTO_INGEST_KNOWLEDGE", "1") != "0":
            self.auto_ingest_knowledge()

    def ingest_document(
        self,
        filename: str,
        content: str,
        source_type: str = "manual",
        source_path: str = "",
    ) -> dict[str, Any]:
        safe_name = Path(filename).name or "untitled.txt"
        return self.store.add_document(
            safe_name,
            content,
            self.chunker,
            source_type=source_type,
            source_path=source_path,
        )

    def auto_ingest_knowledge(self) -> dict[str, Any]:
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        ingested = []
        skipped = []
        for path in sorted(self.knowledge_dir.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            if path.suffix.lower() not in DEFAULT_ALLOWED_EXTENSIONS:
                skipped.append({"path": str(path), "reason": "unsupported_extension"})
                continue
            if path.stat().st_size > MAX_KNOWLEDGE_FILE_BYTES:
                skipped.append({"path": str(path), "reason": "file_too_large"})
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            relative_name = str(path.relative_to(self.knowledge_dir))
            ingested.append(
                self.ingest_document(
                    relative_name,
                    content,
                    source_type="knowledge",
                    source_path=str(path.resolve()),
                )
            )
        return {"ingested": ingested, "skipped": skipped, "status": self.status()}

    def answer(self, question: str, top_k: int = 4, auto_research: bool = True) -> dict[str, Any]:
        retrieval_query = self.memory.enhanced_query(question)
        raw_hits = self.store.search(retrieval_query, top_k=top_k)
        hits = [hit for hit in raw_hits if hit.score > 0.02] or raw_hits[:1]
        if not hits:
            self.memory.observe(question, "Không có dữ liệu để truy xuất.", [])
            return {
                "answer": "Chưa có dữ liệu riêng trong index. Hãy nạp tài liệu trước khi hỏi.",
                "sources": [],
                "mode": self.llm_mode(),
                "profile": self.memory.status(),
                "research": {"updated": False, "reason": "empty_index"},
            }

        answer = self._answer_with_openai(question, hits)
        mode = self.llm_mode()
        if answer is None:
            answer = self._answer_locally(question, hits)
            mode = "local-extractive"
        self.memory.observe(question, answer, hits)

        research = {"updated": False, "reason": "disabled"}
        if auto_research:
            research = self.refresh_research_note(question, hits)

        return {
            "answer": answer,
            "sources": [self._source_payload(index, hit) for index, hit in enumerate(hits, start=1)],
            "mode": mode,
            "profile": self.memory.status(),
            "research": research,
        }

    def refresh_research_note(self, question: str, hits: list[SearchHit]) -> dict[str, Any]:
        if not hits:
            return {"updated": False, "reason": "no_hits"}
        note = self.memory.build_research_note(question, hits)
        result = self.ingest_document(
            "auto-research-memory.md",
            note,
            source_type="memory",
            source_path="chat-memory",
        )
        return {
            "updated": True,
            "document_id": result["document_id"],
            "chunk_count": result["chunk_count"],
        }

    def clear(self, clear_memory: bool = True) -> dict[str, Any]:
        result = self.store.clear()
        if clear_memory:
            self.memory.clear()
        return {"cleared": True, "status": self.status()}

    def status(self) -> dict[str, Any]:
        status = self.store.status()
        status["llm_mode"] = self.llm_mode()
        status["profile"] = self.memory.status()
        status["knowledge_dir"] = str(self.knowledge_dir)
        status["auto_ingest_knowledge"] = os.getenv("AUTO_INGEST_KNOWLEDGE", "1") != "0"
        return status

    def llm_mode(self) -> str:
        if os.getenv("OPENAI_API_KEY"):
            return f"openai:{os.getenv('OPENAI_MODEL', 'configured-model')}"
        return "local-extractive"

    def _answer_locally(self, question: str, hits: list[SearchHit]) -> str:
        structured_answer = self._answer_rag_knowledge(question, hits)
        if structured_answer:
            return structured_answer

        query_tokens = set(tokenize(question))
        candidates: list[tuple[float, str, int]] = []
        for source_index, hit in enumerate(hits, start=1):
            sentences = [part.strip() for part in SENTENCE_RE.split(hit.text) if part.strip()]
            if not sentences:
                sentences = [hit.text.strip()]
            for sentence in sentences:
                sentence_tokens = set(tokenize(sentence))
                overlap = len(query_tokens & sentence_tokens)
                score = hit.score + overlap * 0.08
                candidates.append((score, sentence, source_index))

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected: list[tuple[str, int]] = []
        seen = set()
        for score, sentence, source_index in candidates:
            normalized = strip_diacritics(sentence[:140])
            if normalized in seen:
                continue
            seen.add(normalized)
            selected.append((sentence, source_index))
            if len(selected) == 3:
                break

        if not selected or candidates[0][0] < 0.08:
            return (
                "Mình chưa thấy đoạn dữ liệu đủ khớp để trả lời chắc chắn. "
                "Các nguồn gần nhất đã được hiển thị để bạn kiểm tra lại."
            )

        lines = ["Dựa trên dữ liệu riêng đã nạp:"]
        for sentence, source_index in selected:
            lines.append(f"- {sentence} [S{source_index}]")
        lines.append("Nếu cần câu trả lời tổng hợp tự nhiên hơn, cấu hình OPENAI_API_KEY cho server.")
        return "\n".join(lines)

    def _answer_rag_knowledge(self, question: str, hits: list[SearchHit]) -> str | None:
        question_norm = strip_diacritics(question)
        context_norm = strip_diacritics(" ".join(hit.text for hit in hits))
        if "rag" not in question_norm and "retrieval" not in question_norm and "chatbot" not in question_norm:
            return None
        if not any(term in context_norm for term in ("truy xuat", "tang cuong", "tao sinh", "knowledge base")):
            return None

        cite_definition = self._citation_for(hits, ["retrieval-augmented", "tao tang cuong", "mo hinh ngon ngu"])
        cite_flow = self._citation_for(hits, ["truy xuat", "tang cuong", "tao sinh"])
        cite_benefits = self._citation_for(hits, ["ao tuong", "noi bo", "trich dan", "minh bach"])
        cite_apps = self._citation_for(hits, ["cham soc khach hang", "doanh nghiep", "y te", "phap ly"])
        cite_compare = self._citation_for(hits, ["chatbot truyen thong", "quy tac", "kich ban", "co so tri thuc"])

        wants_flow = any(term in question_norm for term in ("hoat dong", "co che", "quy trinh", "buoc", "van hanh"))
        wants_benefits = any(term in question_norm for term in ("tam quan trong", "loi ich", "ao tuong", "tin cay", "minh bach"))
        wants_apps = any(term in question_norm for term in ("ung dung", "vi du", "dung trong", "doanh nghiep", "khach hang"))
        wants_compare = any(term in question_norm for term in ("khac biet", "so sanh", "truyen thong"))
        wants_knowledge_base = any(term in question_norm for term in ("knowledge base", "co so tri thuc", "kho tai lieu"))

        if wants_compare:
            return "\n".join(
                [
                    "Dựa trên knowledge base đã nạp:",
                    f"- Chatbot truyền thống dựa nhiều vào kịch bản/quy tắc lập trình sẵn, nên nguồn tri thức hẹp và thường cần cập nhật thủ công. {cite_compare}",
                    f"- Chatbot RAG dùng cơ sở tri thức bên ngoài kết hợp LLM, hiểu ngữ cảnh tự nhiên tốt hơn và cập nhật theo tài liệu mới. {cite_compare}",
                    f"- Điểm mạnh của RAG là giảm sai lệch nhờ tra cứu dữ liệu và có thể hiển thị nguồn trích dẫn để kiểm chứng. {cite_benefits}",
                ]
            )

        if wants_flow:
            return "\n".join(
                [
                    "Cơ chế RAG gồm 3 bước:",
                    f"1. Truy xuất: tìm thông tin liên quan từ Knowledge Base khi người dùng đặt câu hỏi. {cite_flow}",
                    f"2. Tăng cường: kết hợp câu hỏi với dữ liệu vừa truy xuất để tạo ngữ cảnh chi tiết. {cite_flow}",
                    f"3. Tạo sinh: LLM dùng ngữ cảnh đã tăng cường để tạo câu trả lời chính xác, dễ hiểu. {cite_flow}",
                ]
            )

        if wants_knowledge_base:
            return "\n".join(
                [
                    "Trong RAG, Knowledge Base là kho tri thức để hệ thống tra cứu trước khi trả lời.",
                    f"- Khi người dùng đặt câu hỏi, bước Truy xuất sẽ tìm thông tin liên quan trong Knowledge Base. {cite_flow}",
                    f"- Dữ liệu truy xuất được kết hợp với câu hỏi để tạo ngữ cảnh chi tiết cho LLM. {cite_flow}",
                    f"- Nhờ vậy chatbot có thể trả lời theo tài liệu nội bộ/tài liệu đáng tin cậy thay vì chỉ dựa vào kiến thức đã huấn luyện. {cite_definition}",
                ]
            )

        if wants_benefits:
            return "\n".join(
                [
                    "Tầm quan trọng của Chatbot RAG:",
                    f"- Giảm ảo tưởng vì LLM phải dựa vào dữ liệu thực tế đã truy xuất. {cite_benefits}",
                    f"- Truy cập dữ liệu nội bộ hoặc gần thời gian thực mà không cần đào tạo lại mô hình. {cite_benefits}",
                    f"- Tăng độ tin cậy và minh bạch vì có thể hiển thị nguồn trích dẫn. {cite_benefits}",
                ]
            )

        if wants_apps:
            return "\n".join(
                [
                    "Một số ứng dụng phù hợp của Chatbot RAG:",
                    f"- Chăm sóc khách hàng: hỏi đáp về bảo hành, giá sản phẩm, hướng dẫn sử dụng. {cite_apps}",
                    f"- Doanh nghiệp: tìm kiếm quy trình nhân sự, chính sách nội bộ, báo cáo tài chính, tài liệu kỹ thuật. {cite_apps}",
                    f"- Y tế/pháp lý: truy xuất hồ sơ bệnh án hoặc văn bản pháp luật phức tạp với yêu cầu kiểm chứng nguồn cao. {cite_apps}",
                ]
            )

        return "\n".join(
            [
                "Chatbot RAG là chatbot AI kết hợp LLM với cơ chế truy xuất dữ liệu từ nguồn bên ngoài trước khi trả lời.",
                f"- Thay vì chỉ dựa vào dữ liệu đã huấn luyện, nó tìm thông tin liên quan từ kho tài liệu đáng tin cậy như cơ sở dữ liệu nội bộ, tài liệu kỹ thuật hoặc trang web. {cite_definition}",
                f"- Sau đó hệ thống đưa dữ liệu truy xuất vào ngữ cảnh để LLM tạo câu trả lời chính xác và cập nhật hơn. {cite_flow}",
                f"- Giá trị chính là AI hóa dữ liệu nội bộ, giảm ảo tưởng và tăng minh bạch nhờ nguồn trích dẫn. {cite_benefits}",
            ]
        )

    @staticmethod
    def _citation_for(hits: list[SearchHit], keywords: list[str]) -> str:
        normalized_keywords = [strip_diacritics(keyword) for keyword in keywords]
        for index, hit in enumerate(hits, start=1):
            text = strip_diacritics(hit.text)
            if any(keyword in text for keyword in normalized_keywords):
                return f"[S{index}]"
        return "[S1]"

    def _answer_with_openai(self, question: str, hits: list[SearchHit]) -> str | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        context = "\n\n".join(f"[S{index}] {hit.text}" for index, hit in enumerate(hits, start=1))
        memory_context = self.memory.prompt_context()
        prompt = (
            "Bạn là chatbot RAG cho dữ liệu nội bộ. Chỉ dùng CONTEXT để trả lời. "
            "Nếu context không đủ, nói rõ là chưa đủ dữ liệu. Trả lời bằng tiếng Việt, "
            "ngắn gọn, và gắn citation dạng [S1], [S2].\n\n"
            f"HỒ SƠ HỘI THOẠI:\n{memory_context}\n\n"
            f"CONTEXT:\n{context}\n\nCÂU HỎI: {question}"
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Bạn trả lời có kiểm chứng bằng nguồn truy xuất."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None
        return data.get("choices", [{}])[0].get("message", {}).get("content")

    @staticmethod
    def _source_payload(index: int, hit: SearchHit) -> dict[str, Any]:
        return {
            "label": f"S{index}",
            "document_name": hit.document_name,
            "chunk_index": hit.index,
            "score": hit.score,
            "text": hit.text,
            "source_type": hit.source_type,
        }


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
