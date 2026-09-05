import logging
import time
from contextlib import contextmanager
from app.schemas import Analysis, Explanations, Rejected, Success, Score, Recommendation
from app.errors import AppError

log = logging.getLogger("outfit")


@contextmanager
def stage(timings, name):
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = round((time.perf_counter() - start) * 1000, 2)


class Pipeline:
    def __init__(self, vlm, encoder, catalog, settings):
        self.vlm, self.encoder, self.catalog, self.settings = vlm, encoder, catalog, settings

    def analyze(self, image, request_id, deadline, timings):
        started = time.perf_counter()

        def check_analysis(a):
            if a.status == "ok":
                mode = (
                    "similar_alternative"
                    if a.aesthetic_score >= self.settings.high_score_similarity_threshold
                    else "improve"
                )
                if a.replacement_mode != mode:
                    raise ValueError("Wrong replacement mode")

        try:
            with stage(timings, "vlm_analysis_ms"):
                a = self.vlm.call(
                    "outfit_analysis_v1.txt",
                    [("ORIGINAL OUTFIT", image)],
                    {"high_score_similarity_threshold": self.settings.high_score_similarity_threshold},
                    Analysis,
                    deadline,
                    check_analysis,
                )
            if a.status == "rejected":
                code = a.scene.error_code if not a.scene.valid else "NOT_ENOUGH_ITEMS"
                message = (
                    "Không đủ item để đánh giá."
                    if code == "NOT_ENOUGH_ITEMS"
                    else "Ảnh chứa nhiều người hoặc trang phục gây nhầm lẫn. Vui lòng tải ảnh chỉ có một outfit chính."
                )
                return Rejected(
                    error_code=code,
                    message_vi=message,
                    counted_item_count=a.counted_item_count,
                    garments=a.garments,
                )
            with stage(timings, "fashionclip_query_embedding_ms"):
                vectors = self.encoder.encode(a.replacement_queries_en)
            with stage(timings, "retrieval_ms"):
                candidates = self.catalog.retrieve(
                    a.replacement_queries_en, vectors, a.problematic_item.category
                )
                images = []
                try:
                    for c in candidates:
                        images.append(
                            (
                                f"CANDIDATE {c['rank']} item_id={c['item_id']}",
                                self.catalog.image(c["item_id"]),
                            )
                        )
                except (ValueError, OSError, KeyError) as exc:
                    for _, im in images:
                        im.close()
                    raise AppError(
                        "RETRIEVAL_ERROR", "Không thể đọc ảnh sản phẩm thực. Vui lòng thử lại."
                    ) from exc

            def check_reasons(e):
                if [r.item_id for r in e.recommendations] != [c["item_id"] for c in candidates]:
                    raise ValueError("Explanation ID differs from retrieved image")

            try:
                with stage(timings, "vlm_explanation_ms"):
                    e = self.vlm.call(
                        "recommendation_explanation_v1.txt",
                        [("ORIGINAL OUTFIT", image), *images],
                        {"analysis": a.model_dump(mode="json"), "candidates": candidates},
                        Explanations,
                        deadline,
                        check_reasons,
                    )
            finally:
                for _, im in images:
                    im.close()
            recommendations = [
                Recommendation(
                    **r.model_dump(),
                    category=c["category"],
                    query_en=c["query_en"],
                    image_url="/api/items/" + c["item_id"] + "/image",
                )
                for r, c in zip(e.recommendations, candidates)
            ]
            if a.replacement_mode == "similar_alternative":
                intro = f"Outfit hiện tại đã có độ hài hòa cao. Hệ thống chọn {a.problematic_item.name_vi} để thử các biến thể tương tự. Đây là những lựa chọn để đổi phong cách, không phải khẳng định outfit hiện tại có lỗi."
            else:
                intro = f"Trong outfit này, {a.problematic_item.name_vi} là món được ưu tiên cân nhắc thay. {a.problematic_item.reason_vi}"
            commentary = (
                intro
                + "\n\n"
                + "\n\n".join(f"{r.rank}. {r.display_name_vi}: {r.reason_vi}" for r in recommendations)
            )
            return Success(
                garments=a.garments,
                score=Score(total=a.aesthetic_score, dimensions=a.rubric),
                problematic_item=a.problematic_item,
                replacement_mode=a.replacement_mode,
                recommendations=recommendations,
                commentary_vi=commentary,
            )
        finally:
            timings["total_ms"] = round(
                (time.perf_counter() - started) * 1000 + timings.get("image_validation_ms", 0), 2
            )
            log.info("pipeline", extra={"request_id": request_id, "timings": timings})
