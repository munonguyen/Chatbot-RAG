from __future__ import annotations

import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from rag.pipeline import RAGService


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
DATA_DIR = ROOT / "data"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(ROOT / ".env")
service = RAGService(DATA_DIR / "vector_store.json", ROOT / "knowledge")


class RAGRequestHandler(BaseHTTPRequestHandler):
    server_version = "ChatbotRAG/1.0"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json({"ok": True})
            return
        if path == "/api/status":
            self._send_json(service.status())
            return
        if path in {"/", "/index.html"}:
            self._send_file(STATIC_DIR / "index.html")
            return
        if path.startswith("/static/"):
            relative = unquote(path.removeprefix("/static/"))
            self._send_file(STATIC_DIR / relative)
            return
        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/documents":
                result = self._handle_documents(payload)
            elif path == "/api/auto-ingest":
                result = service.auto_ingest_knowledge()
            elif path == "/api/chat":
                result = self._handle_chat(payload)
            elif path == "/api/clear":
                result = service.clear(clear_memory=bool(payload.get("clear_memory", True)))
            else:
                self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(result)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # Keeps the demo server responsive during bad input.
            self._send_json({"error": f"Server error: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:
        if os.getenv("RAG_DEBUG"):
            super().log_message(format, *args)

    def _handle_documents(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_documents = payload.get("documents")
        if raw_documents is None:
            raw_documents = [{"filename": payload.get("filename"), "content": payload.get("content")}]
        if not isinstance(raw_documents, list):
            raise ValueError("documents must be a list")

        ingested = []
        for item in raw_documents:
            if not isinstance(item, dict):
                raise ValueError("each document must be an object")
            filename = str(item.get("filename") or "untitled.txt").strip()
            content = str(item.get("content") or "")
            ingested.append(service.ingest_document(filename, content))
        return {"ingested": ingested, "status": service.status()}

    def _handle_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        message = str(payload.get("message") or "").strip()
        top_k = int(payload.get("top_k") or 4)
        auto_research = bool(payload.get("auto_research", True))
        if not message:
            raise ValueError("message is required")
        return service.answer(message, top_k=top_k, auto_research=auto_research)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > 10 * 1024 * 1024:
            raise ValueError("request body is too large")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSON body") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        resolved = path.resolve()
        if not self._is_under(resolved, STATIC_DIR.resolve()) or not resolved.is_file():
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        body = resolved.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _is_under(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "127.0.0.1")
    httpd = ThreadingHTTPServer((host, port), RAGRequestHandler)
    print(f"Chatbot RAG running at http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
