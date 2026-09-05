import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

type Dimension = { score: number; reason_vi: string };
type Garment = {
  garment_id: string;
  category: string;
  name_vi: string;
  name_en: string;
  description_en: string;
};
type Recommendation = {
  rank: number;
  item_id: string;
  category: string;
  image_url: string;
  display_name_vi: string;
  reason_vi: string;
  reason_facets: string[];
};
type Result = {
  status: "ok";
  garments: Garment[];
  score: { total: number; dimensions: Record<string, Dimension> };
  problematic_item: { category: string; name_vi: string; reason_vi: string };
  replacement_mode: "improve" | "similar_alternative";
  recommendations: Recommendation[];
  commentary_vi: string;
};
type Failure = {
  status: "error" | "rejected";
  error_code: string;
  message_vi: string;
  request_id?: string;
};
const labels: Record<string, string> = {
  color_harmony: "Hài hòa màu sắc",
  style_coherence: "Nhất quán phong cách",
  silhouette_proportion: "Phom dáng & tỷ lệ",
  formality_occasion_coherence: "Đồng nhất độ trang trọng",
  overall_styling: "Hoàn thiện tổng thể",
};
const errors: Record<string, [string, string]> = {
  NOT_ENOUGH_ITEMS: [
    "Không đủ item để đánh giá.",
    "Hệ thống cần ít nhất 3 món thuộc Top, Bottom, Outerwear, Shoes, Bag, Dress hoặc Hat.",
  ],
  MULTIPLE_PEOPLE: [
    "Ảnh chứa nhiều người.",
    "Vui lòng chọn ảnh chỉ có một người và một outfit chính.",
  ],
  MANNEQUIN_OR_BACKGROUND_OUTFIT: [
    "Có trang phục khác gây nhầm lẫn.",
    "Chọn ảnh không có mannequin hoặc outfit khác ở phía sau.",
  ],
  AMBIGUOUS_SCENE: [
    "Chưa xác định rõ outfit chính.",
    "Chọn ảnh một người, trang phục rõ nét và hậu cảnh đơn giản.",
  ],
  INVALID_IMAGE: [
    "Ảnh chưa hợp lệ.",
    "Chọn một ảnh JPG, PNG hoặc WebP, tối đa 10 MB.",
  ],
  VLM_OUTPUT_ERROR: [
    "Chưa tạo được kết quả hợp lệ.",
    "Bạn có thể thử lại hoặc chọn ảnh rõ hơn.",
  ],
  RETRIEVAL_ERROR: [
    "Chưa tìm đủ ba lựa chọn.",
    "Dữ liệu sản phẩm có thể chưa sẵn sàng. Vui lòng thử lại sau.",
  ],
  MODEL_NOT_READY: [
    "Hệ thống đang chuẩn bị.",
    "Lần khởi động đầu tiên có thể mất vài phút. Vui lòng thử lại sau.",
  ],
  REQUEST_TIMEOUT: [
    "Phân tích mất nhiều thời gian.",
    "Vui lòng chờ một chút rồi thử lại.",
  ],
  SERVER_BUSY: [
    "Hệ thống đang xử lý ảnh khác.",
    "Vui lòng thử lại sau ít phút.",
  ],
  NETWORK_ERROR: [
    "Không kết nối được hệ thống.",
    "Kiểm tra kết nối mạng rồi thử lại.",
  ],
};
function App() {
  const [file, setFile] = useState<File | null>(null),
    [preview, setPreview] = useState(""),
    [busy, setBusy] = useState(false),
    [result, setResult] = useState<Result | null>(null),
    [failure, setFailure] = useState<Failure | null>(null),
    [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const controller = useRef<AbortController | null>(null);
  const output = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => () => controller.current?.abort(), []);
  function select(f?: File) {
    if (busy) return;
    setResult(null);
    setFailure(null);
    if (!f) return;
    if (
      !["image/jpeg", "image/png", "image/webp"].includes(f.type) ||
      f.size > 10 * 1024 * 1024
    ) {
      setFailure({
        status: "error",
        error_code: "INVALID_IMAGE",
        message_vi: "",
      });
      return;
    }
    setFile(f);
  }
  function clear() {
    if (busy) return;
    setFile(null);
    setResult(null);
    setFailure(null);
    if (input.current) input.current.value = "";
  }
  async function analyze() {
    if (!file || busy) return;
    setBusy(true);
    setFailure(null);
    setResult(null);
    const abort = new AbortController();
    controller.current = abort;
    const timer = window.setTimeout(() => abort.abort(), 210000);
    try {
      const body = new FormData();
      body.append("image", file);
      const response = await fetch("/api/analyze", {
        method: "POST",
        body,
        signal: abort.signal,
      });
      let data: Result | Failure;
      try {
        data = await response.json();
      } catch {
        throw new Error("NETWORK_ERROR");
      }
      if (response.ok && data.status === "ok") {
        if (
          !Array.isArray(data.recommendations) ||
          data.recommendations.length !== 3
        )
          throw new Error("VLM_OUTPUT_ERROR");
        setResult(data);
      } else if (data.status === "error" || data.status === "rejected") {
        setFailure(data);
      } else throw new Error("NETWORK_ERROR");
    } catch (e) {
      const code =
        e instanceof DOMException && e.name === "AbortError"
          ? "REQUEST_TIMEOUT"
          : e instanceof Error && e.message === "VLM_OUTPUT_ERROR"
            ? "VLM_OUTPUT_ERROR"
            : "NETWORK_ERROR";
      setFailure({ status: "error", error_code: code, message_vi: "" });
    } finally {
      window.clearTimeout(timer);
      setBusy(false);
      controller.current = null;
      window.setTimeout(
        () =>
          output.current?.scrollIntoView({
            behavior: "smooth",
            block: "nearest",
          }),
        100,
      );
    }
  }
  const similar = result?.replacement_mode === "similar_alternative";
  return (
    <main>
      <header>
        <a className="brand" href="/" aria-label="Outfit Advisor trang chủ">

          <span>
            OUTFIT ADVISOR <span className="muted">· VLM</span>
          </span>
        </a>
        <span className="model-tag">Qwen3-VL · FashionCLIP</span>
      </header>
      <section className="hero">

        <h1>
          Gợi ý thay
          <br className="mobile-break" /> <span>món đồ.</span>
        </h1>
        <p>
          Tải một ảnh outfit lên để hệ thống đánh giá và đề xuất ba lựa chọn
          thay thế phù hợp.
        </p>
      </section>
      <section className="workspace" aria-label="Tải ảnh và phân tích">
        <div
          className={`upload panel ${drag ? "drag" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            if (!busy) setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            select(e.dataTransfer.files[0]);
          }}
        >
          <input
            ref={input}
            id="outfit-file"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            disabled={busy}
            onChange={(e) => select(e.target.files?.[0])}
          />
          {preview ? (
            <>
              <img
                className="outfit-image"
                src={preview}
                alt="Ảnh outfit bạn đã chọn"
              />
              <button
                className="change"
                disabled={busy}
                onClick={() => input.current?.click()}
              >
                Đổi ảnh
              </button>
            </>
          ) : (
            <label className="dropzone" htmlFor="outfit-file">
              <span className="upload-icon" aria-hidden="true">
                ↑
              </span>
              <h2>Outfit của bạn</h2>
              <p>
                Kéo ảnh vào đây hoặc <span>chọn từ thiết bị</span>
              </p>
              <small>JPG, PNG, WebP · Tối đa 10 MB</small>
            </label>
          )}
        </div>
        <div className="analysis panel">
          <span className="step-label">01 / KHÁM PHÁ CÁCH PHỐI</span>
          <h2>Phân tích outfit</h2>
          <p>
            Một góc nhìn về màu sắc, phong cách và tỷ lệ — cùng ba món đồ thực
            để bạn cân nhắc.
          </p>
          <div className="guide">
            <span>Ảnh phù hợp</span>
            <p>
              Một người, ít nhất 3 món đồ chính. Tránh mannequin hoặc trang phục
              khác ở phía sau.
            </p>
          </div>
          <div className="actions">
            <button
              className="primary"
              onClick={analyze}
              disabled={!file || busy}
            >
              {busy ? (
                <>
                  <span className="spinner" />
                  Đang phân tích outfit...
                </>
              ) : (
                "Phân tích outfit"
              )}
              {!busy && <span aria-hidden="true">↗</span>}
            </button>
            <button
              className="secondary"
              onClick={clear}
              disabled={!file || busy}
            >
              Xóa ảnh
            </button>
          </div>
          <p className="warmup">
            Lần khởi động đầu tiên có thể mất vài phút để chuẩn bị mô hình. Ảnh
            tải lên không được lưu lâu dài.
          </p>
        </div>
      </section>
      <div ref={output} aria-live="polite" aria-busy={busy}>
        {busy && (
          <div className="loading panel">
            <span className="spinner" />
            <div>
              <strong>Đang phân tích outfit...</strong>
              <p>
                Đang xem ảnh và tìm các lựa chọn phù hợp. Bạn hãy giữ trang này
                mở.
              </p>
            </div>
          </div>
        )}
        {failure && (
          <section className="error panel" role="alert">
            <span className="error-mark">!</span>
            <div>
              <h2>
                {
                  (errors[failure.error_code] || [
                    "Chưa thể hoàn tất phân tích.",
                    "Vui lòng thử lại sau.",
                  ])[0]
                }
              </h2>
              <p>
                {(errors[failure.error_code] || ["", failure.message_vi])[1]}
              </p>
              {failure.request_id && (
                <small>Mã hỗ trợ: {failure.request_id.slice(0, 12)}</small>
              )}
            </div>
          </section>
        )}
        {result && (
          <div className="results">
            <section className="score-panel panel">
              <div className="score-total">
                <span className="step-label">02 / GÓC NHÌN TỔNG THỂ</span>
                <div className="score-number">
                  {result.score.total}
                  <span>/100</span>
                </div>
                <p>Điểm đánh giá theo rubric của hệ thống</p>
                <small>
                  Đây là nhận xét theo mô hình, không phải tiêu chuẩn thời trang
                  khách quan.
                </small>
              </div>
              <div className="dimensions">
                {Object.entries(labels).map(([key, label]) => {
                  const d = result.score.dimensions[key];
                  return (
                    <div className="dimension" key={key}>
                      <div>
                        <span>{label}</span>
                        <strong>
                          {d.score}
                          <span>/20</span>
                        </strong>
                      </div>
                      <progress max={20} value={d.score} aria-label={label} />
                      <p>{d.reason_vi}</p>
                    </div>
                  );
                })}
              </div>
            </section>
            <div className="suggestions panel">
            <section className="recommendation-layout">
              <aside>
                <h2 className="section-title">
                  {similar ? "MÓN ĐỂ THỬ BIẾN THỂ" : "MÓN CẦN THAY"}
                </h2>
                <div className="selected panel">
                  <span className="category">
                    {result.problematic_item.category}
                  </span>
                  <h3>{result.problematic_item.name_vi}</h3>
                  <p>{result.problematic_item.reason_vi}</p>
                  <span className="mode-label">
                    {similar ? "Biến thể tương tự" : "Gợi ý cải thiện"}
                  </span>
                </div>
              </aside>
              <div>
                <h2 className="section-title">TOP 3 LỰA CHỌN</h2>
                <div className="cards">
                  {result.recommendations.map((item) => (
                    <article
                      className="recommendation panel"
                      key={item.item_id}
                    >
                      <div className="product-image">
                        <img
                          src={item.image_url}
                          alt={item.display_name_vi}
                          loading="lazy"
                        />
                        <span className="rank">0{item.rank}</span>
                      </div>
                      <div className="card-body">
                        <span className="item-category">{item.category}</span>
                        <h3>{item.display_name_vi}</h3>
                        <p>{item.reason_vi}</p>
                        <div className="facets">
                          {item.reason_facets.map((f) => (
                            <span key={f}>{labels[f]}</span>
                          ))}
                        </div>
                      </div>
                    </article>
                  ))}
                </div>
              </div>
            </section>
            <section className="commentary panel">
              <h2 className="section-title">NHẬN XÉT</h2>
              <p>{result.commentary_vi}</p>
            </section>
            </div>
          </div>
        )}
      </div>
      <footer>
        <span>OUTFIT ADVISOR</span>
        <span>Gợi ý để tham khảo. Phong cách là của bạn.</span>
      </footer>
    </main>
  );
}
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
