from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag.pipeline import RAGService, TextChunker


RAG_FOUNDATION = """
Chatbot RAG, viết tắt của Retrieval-Augmented Generation hoặc Tạo tăng cường truy xuất, là một ứng dụng AI hội thoại kết hợp mô hình ngôn ngữ lớn (LLM) với khả năng truy xuất dữ liệu từ các nguồn bên ngoài.

Thay vì chỉ dựa vào dữ liệu đã được đào tạo, chatbot RAG tìm kiếm thông tin liên quan từ kho tài liệu đáng tin cậy, ví dụ cơ sở dữ liệu nội bộ, tài liệu kỹ thuật hoặc trang web, trước khi tạo câu trả lời.

Quy trình hoạt động của RAG gồm 3 bước chính: Truy xuất, Tăng cường và Tạo sinh. Truy xuất là tìm thông tin liên quan từ Knowledge Base. Tăng cường là kết hợp câu hỏi với dữ liệu truy xuất thành ngữ cảnh chi tiết. Tạo sinh là LLM dựa trên ngữ cảnh để tạo câu trả lời chính xác.

RAG giúp giảm thiểu ảo tưởng, truy cập dữ liệu nội bộ hoặc gần thời gian thực, và tăng độ tin cậy minh bạch nhờ nguồn trích dẫn.

Ứng dụng gồm chăm sóc khách hàng, doanh nghiệp, y tế và pháp lý.

Chatbot truyền thống dựa vào kịch bản lập trình sẵn hoặc quy tắc cố định. Chatbot RAG dùng cơ sở tri thức bên ngoài kết hợp với LLM, hiểu ngữ cảnh tự nhiên tốt hơn và cập nhật theo tài liệu mới.
"""


class RAGPipelineTest(unittest.TestCase):
    def test_chunker_keeps_content_and_splits_long_text(self) -> None:
        chunker = TextChunker(max_words=20, overlap_words=5)
        text = " ".join(f"word{i}" for i in range(70))

        chunks = chunker.split(text)

        self.assertGreater(len(chunks), 1)
        self.assertIn("word0", chunks[0])
        self.assertTrue(any("word69" in chunk for chunk in chunks))

    def test_service_ingests_and_retrieves_private_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = RAGService(Path(directory) / "store.json")
            service.ingest_document(
                "policy.txt",
                "Gói Enterprise có hỗ trợ ưu tiên 24/7 và thời gian phản hồi mục tiêu 60 phút.",
            )

            result = service.answer("Enterprise được hỗ trợ trong bao lâu?")

            self.assertEqual(len(result["sources"]), 1)
            self.assertIn("Enterprise", result["answer"])
            self.assertIn("60 phút", result["answer"])

    def test_auto_ingest_and_chat_memory_build_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            knowledge = root / "knowledge"
            knowledge.mkdir()
            (knowledge / "rag-guide.md").write_text(
                "RAG dùng embedding, vector store và semantic retrieval để trả lời theo dữ liệu riêng.",
                encoding="utf-8",
            )
            service = RAGService(root / "data" / "store.json", knowledge)

            status = service.status()
            self.assertEqual(status["document_count"], 1)

            result = service.answer("Chatbot RAG dùng vector store để làm gì?")

            self.assertTrue(result["research"]["updated"])
            self.assertEqual(result["profile"]["domain"], "GenAI/RAG")
            self.assertGreaterEqual(service.status()["document_count"], 2)

    def test_rag_knowledge_definition_answer_is_structured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = RAGService(Path(directory) / "store.json")
            service.ingest_document("rag-foundation.md", RAG_FOUNDATION)

            result = service.answer("Chatbot RAG là gì?", auto_research=False)

            self.assertIn("LLM", result["answer"])
            self.assertIn("nguồn bên ngoài", result["answer"])
            self.assertIn("[S1]", result["answer"])

    def test_rag_knowledge_flow_answer_has_three_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = RAGService(Path(directory) / "store.json")
            service.ingest_document("rag-foundation.md", RAG_FOUNDATION)

            result = service.answer("Cơ chế hoạt động của RAG gồm những bước nào?", auto_research=False)

            self.assertIn("1. Truy xuất", result["answer"])
            self.assertIn("2. Tăng cường", result["answer"])
            self.assertIn("3. Tạo sinh", result["answer"])

    def test_rag_knowledge_benefits_apps_and_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = RAGService(Path(directory) / "store.json")
            service.ingest_document("rag-foundation.md", RAG_FOUNDATION)

            benefits = service.answer("Tầm quan trọng của Chatbot RAG là gì?", auto_research=False)
            apps = service.answer("Ví dụ ứng dụng RAG trong doanh nghiệp?", auto_research=False)
            compare = service.answer("So sánh chatbot truyền thống và chatbot RAG", auto_research=False)

            self.assertIn("Giảm ảo tưởng", benefits["answer"])
            self.assertIn("Doanh nghiệp", apps["answer"])
            self.assertIn("Chatbot truyền thống", compare["answer"])
            self.assertIn("Chatbot RAG", compare["answer"])


if __name__ == "__main__":
    unittest.main()
