const state = {
  files: [],
  busy: false,
};

const els = {
  fileInput: document.querySelector("#fileInput"),
  ingestButton: document.querySelector("#ingestButton"),
  autoIngestButton: document.querySelector("#autoIngestButton"),
  sampleButton: document.querySelector("#sampleButton"),
  clearButton: document.querySelector("#clearButton"),
  docCount: document.querySelector("#docCount"),
  chunkCount: document.querySelector("#chunkCount"),
  llmMode: document.querySelector("#llmMode"),
  domainName: document.querySelector("#domainName"),
  domainTerms: document.querySelector("#domainTerms"),
  documentList: document.querySelector("#documentList"),
  chatStream: document.querySelector("#chatStream"),
  chatForm: document.querySelector("#chatForm"),
  messageInput: document.querySelector("#messageInput"),
  autoResearchToggle: document.querySelector("#autoResearchToggle"),
  sourceList: document.querySelector("#sourceList"),
  statusPill: document.querySelector("#statusPill"),
};

const sampleDocuments = [
  {
    filename: "company-policy.md",
    content: `Chính sách hỗ trợ khách hàng

Khách hàng gói Business được phản hồi trong vòng 4 giờ làm việc. Khách hàng gói Enterprise có kênh ưu tiên 24/7 và thời gian phản hồi mục tiêu là 60 phút.

Quy trình hoàn tiền chỉ áp dụng trong 14 ngày đầu kể từ ngày kích hoạt. Yêu cầu hoàn tiền cần có mã đơn hàng, email đăng ký và lý do hủy dịch vụ.

Dữ liệu nội bộ không được gửi ra ngoài hệ thống nếu chưa có phê duyệt của bộ phận bảo mật.`,
  },
  {
    filename: "product-faq.txt",
    content: `Câu hỏi sản phẩm

Chatbot RAG dùng dữ liệu riêng bằng cách tách tài liệu thành các đoạn nhỏ, tạo embedding cho từng đoạn, lưu vào vector store, sau đó truy xuất các đoạn liên quan khi người dùng hỏi.

Ưu điểm chính là câu trả lời bám vào tài liệu doanh nghiệp thay vì chỉ dựa vào kiến thức chung của mô hình ngôn ngữ.

Khi dữ liệu thay đổi, chỉ cần nạp lại tài liệu liên quan để tạo index mới.`,
  },
];

function setBusy(value, label = "Đang xử lý") {
  state.busy = value;
  els.ingestButton.disabled = value;
  els.autoIngestButton.disabled = value;
  els.sampleButton.disabled = value;
  els.clearButton.disabled = value;
  els.messageInput.disabled = value;
  els.statusPill.textContent = value ? label : "Sẵn sàng";
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

async function refreshStatus() {
  const status = await api("/api/status");
  els.docCount.textContent = status.document_count;
  els.chunkCount.textContent = status.chunk_count;
  els.llmMode.textContent = status.llm_mode.includes("openai") ? "OpenAI" : "Local";
  els.domainName.textContent = status.profile?.domain || "Đang học từ hội thoại";
  const terms = status.profile?.top_terms || [];
  els.domainTerms.textContent = terms.length ? terms.slice(0, 8).join(", ") : "Chưa có từ khóa.";
  renderDocuments(status.documents || []);
}

function renderDocuments(documents) {
  if (!documents.length) {
    els.documentList.innerHTML = `<p class="empty-copy">Index đang trống.</p>`;
    return;
  }
  els.documentList.innerHTML = documents
    .map(
      (doc) => `
        <article class="document-item">
          <strong>${escapeHtml(doc.name)}</strong>
          <span>${doc.chunk_count} chunks · ${sourceTypeLabel(doc.source_type)}</span>
        </article>
      `,
    )
    .join("");
}

function renderSources(sources) {
  if (!sources || !sources.length) {
    els.sourceList.innerHTML = `<p class="empty-copy">Chưa có nguồn truy xuất.</p>`;
    return;
  }
  els.sourceList.innerHTML = sources
    .map(
      (source) => `
        <article class="source-item">
          <strong>${source.label} · ${escapeHtml(source.document_name)}</strong>
          <div class="source-meta">Chunk ${source.chunk_index + 1} · ${sourceTypeLabel(source.source_type)}</div>
          <p>${escapeHtml(trimText(source.text, 460))}</p>
          <span class="source-score">Score ${Number(source.score).toFixed(3)}</span>
        </article>
      `,
    )
    .join("");
}

function addMessage(role, text) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "Bạn" : "AI";
  const body = document.createElement("p");
  body.textContent = text;
  article.append(avatar, body);
  els.chatStream.append(article);
  els.chatStream.scrollTop = els.chatStream.scrollHeight;
}

async function readSelectedFiles() {
  const files = Array.from(els.fileInput.files || []);
  if (!files.length) {
    throw new Error("Chưa chọn tài liệu.");
  }
  return Promise.all(
    files.map(async (file) => ({
      filename: file.name,
      content: await file.text(),
    })),
  );
}

async function ingestDocuments(documents) {
  setBusy(true, "Đang index");
  try {
    await api("/api/documents", {
      method: "POST",
      body: JSON.stringify({ documents }),
    });
    await refreshStatus();
    els.fileInput.value = "";
  } finally {
    setBusy(false);
  }
}

els.ingestButton.addEventListener("click", async () => {
  try {
    const documents = await readSelectedFiles();
    await ingestDocuments(documents);
  } catch (error) {
    addMessage("assistant", error.message);
  }
});

els.sampleButton.addEventListener("click", async () => {
  try {
    await ingestDocuments(sampleDocuments);
  } catch (error) {
    addMessage("assistant", error.message);
  }
});

els.autoIngestButton.addEventListener("click", async () => {
  setBusy(true, "Đang quét");
  try {
    const result = await api("/api/auto-ingest", { method: "POST", body: "{}" });
    await refreshStatus();
    addMessage(
      "assistant",
      `Đã quét thư mục knowledge: ${result.ingested.length} tài liệu được nạp, ${result.skipped.length} file bị bỏ qua.`,
    );
  } catch (error) {
    addMessage("assistant", error.message);
  } finally {
    setBusy(false);
  }
});

els.clearButton.addEventListener("click", async () => {
  setBusy(true, "Đang xóa");
  try {
    await api("/api/clear", { method: "POST", body: "{}" });
    renderSources([]);
    await refreshStatus();
  } catch (error) {
    addMessage("assistant", error.message);
  } finally {
    setBusy(false);
  }
});

els.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = els.messageInput.value.trim();
  if (!message || state.busy) return;

  els.messageInput.value = "";
  addMessage("user", message);
  setBusy(true, "Đang truy xuất");
  try {
    const result = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        message,
        top_k: 4,
        auto_research: els.autoResearchToggle.checked,
      }),
    });
    addMessage("assistant", result.answer);
    renderSources(result.sources);
    await refreshStatus();
  } catch (error) {
    addMessage("assistant", error.message);
  } finally {
    setBusy(false);
    els.messageInput.focus();
  }
});

function trimText(text, maxLength) {
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength - 1)}…`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function sourceTypeLabel(sourceType) {
  const labels = {
    manual: "Thủ công",
    knowledge: "Knowledge",
    memory: "Memory",
  };
  return labels[sourceType] || sourceType || "Không rõ";
}

refreshStatus().catch((error) => {
  addMessage("assistant", error.message);
});
