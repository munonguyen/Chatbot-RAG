# Chatbot RAG dùng dữ liệu riêng

Prototype này bám theo mô hình trong ảnh: nạp tài liệu nội bộ, tách thành chunks, tạo embedding, lưu vào vector store, truy xuất context theo câu hỏi và sinh câu trả lời có nguồn.

## Luồng vận hành

1. **Loading & Indexing**: người dùng nạp `.txt`, `.md`, `.csv`, `.json`, `.log`; backend làm sạch nội dung và tách thành chunks.
2. **Embedding & Vector Store**: mỗi chunk được chuyển thành vector bằng local hashing embedder và lưu tại `data/vector_store.json`.
3. **Query Processing**: câu hỏi được embed cùng cách với tài liệu.
4. **Semantic Search**: hệ thống tính cosine similarity, cộng điểm lexical overlap để lấy các chunk liên quan nhất.
5. **Answering**: mặc định trả lời kiểu extractive local để không cần API key. Nếu có `OPENAI_API_KEY`, backend dùng LLM để tổng hợp câu trả lời từ context và gắn citation `[S1]`.

## Tự động nạp và tự nghiên cứu

- Đặt tài liệu chuyên ngành vào thư mục `knowledge/`. Khi server khởi động, app tự quét thư mục này và index các file `.txt`, `.md`, `.csv`, `.json`, `.log`.
- Repo đã có sẵn `knowledge/rag-foundation.md`, chứa kiến thức nền về định nghĩa RAG, cơ chế 3 bước, lợi ích, ứng dụng và so sánh với chatbot truyền thống.
- Trong giao diện, nút **Quét thư mục** gọi lại quá trình auto-ingest mà không cần restart server.
- Mỗi câu hỏi được lưu vào `data/chat_memory.json`. Hệ thống suy luận chuyên ngành đang trao đổi, trích từ khóa nổi bật, và dùng lịch sử gần đây để mở rộng truy vấn retrieval.
- Khi bật **Tự nghiên cứu**, sau mỗi câu hỏi app tạo/cập nhật tài liệu `auto-research-memory.md` trong vector store. Tài liệu này là ghi chú nghiên cứu có nguồn từ lịch sử chat và các đoạn context đã truy xuất.
- Cơ chế này không tự bịa kiến thức mới. Nếu muốn nghiên cứu từ internet hoặc kho tài liệu doanh nghiệp, hãy đưa nguồn đó vào `knowledge/` hoặc tích hợp thêm connector/search API.

## Chạy local

```bash
python3 app.py
```

Mở `http://127.0.0.1:8000`.

## Tùy chọn dùng LLM

```bash
cp .env.example .env
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4.1-mini"
python3 app.py
```

Không có key thì app vẫn chạy bằng chế độ `local-extractive`.

## Kiểm tra

```bash
python3 -m unittest
```

## Cấu trúc

- `app.py`: HTTP server, static web, API ingest/chat/status.
- `rag/pipeline.py`: chunking, embedding, vector store, retrieval, answer generation.
- `static/`: giao diện chatbot và quản lý knowledge base.
- `tests/`: kiểm tra pipeline lõi.
