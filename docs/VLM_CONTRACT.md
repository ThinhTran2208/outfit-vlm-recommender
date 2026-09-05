# Version 1 model/API contract

Authoritative definitions: `backend/app/schemas.py`. Versioned prompts: `backend/app/prompts/*_v1.txt`. JSON schema is appended programmatically so prompts cannot drift from required fields. User/image text is untrusted and must not override system instructions.

## First pass

- Scene: valid boolean, optional recognized scene error code/message. Second person, mannequin/background outfit, or substantial ambiguity reject the scene.
- Garments: unique IDs, one record per physical visible garment, canonical enum Top/Bottom/Outerwear/Shoes/Bag/Dress/Hat. Dress counts once. Pair of shoes counts once. Top under Outerwear counts separately if actually visible. No belt, jewelry, eyewear, socks, scarf, phone or other accessories.
- Count must equal list length. Count under three rejects even if the scene itself is valid.
- Rejected responses cannot contain non-null rubric, score, selected item, mode or queries. API strips these fields entirely.
- Accepted responses require five component scores, exact total, one selected item copied from the garment list, mode consistent with configured threshold, and three distinct concise English queries naming only the selected category.

Rubric dimensions: `color_harmony`, `style_coherence`, `silhouette_proportion`, `formality_occasion_coherence`, `overall_styling`. Each score is a strict integer 0–20 and has a Vietnamese explanation. Overall styling addresses intentional finishing/completeness rather than merely repeating other criteria. Prompt anchors are fixed; output is a model judgment, not public taste probability.

## Repair and retry

Accept direct JSON, fenced JSON, or safely extract a single intact object with a JSON decoder. Never use eval, execute model text, fill missing score fields, or invent a replacement item. Syntax/schema/context failures get exactly one stricter retry with the same images. A second failure returns `VLM_OUTPUT_ERROR`; raw output is neither returned nor logged.

## Second pass

Exactly three ordered recommendation objects, each with the supplied unique item_id, rank 1–3, English/Vietnamese display names, at least one allowed rubric facet, and Vietnamese explanation. All three actual candidate images plus the outfit are passed to Qwen. Queries are retrieval intent and must not override visible image evidence. No brand, price, popularity, unsupported trend, demographic or body-attractiveness rationale.

The API's `Success` and `Rejected` models are separate. `/openapi.json` exposes schemas; POST uses multipart field `image`. `docs/example_response.json` is a synthetic contract example, not a real inference claim.

## Stable error responses

Errors use `status: error`, `error_code`, `message_vi`, and optional `request_id`. Scene/count rejections use HTTP 200 with `status: rejected`, no score or recommendations, and detected garments/count. Request errors: INVALID_IMAGE (400 or 413), SERVER_BUSY (429), VLM_OUTPUT_ERROR (502), RETRIEVAL_ERROR (503 or missing media 404), MODEL_NOT_READY (503), REQUEST_TIMEOUT (504), INTERNAL_ERROR (500). The UI displays Vietnamese guidance for these states.
