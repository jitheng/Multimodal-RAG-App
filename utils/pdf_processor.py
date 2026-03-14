"""
PDF processing utilities — optimised for scanned/image-based PDFs.

Strategy:
  1. Convert each PDF page to a PNG image (via pdf2image / poppler).
  2. Send each page image to GPT-4o vision, which returns structured JSON with:
       - page_type: "text" | "figure" | "table" | "mixed"
       - text_content: full OCR'd text from the page
       - tables: list of markdown tables found on the page
       - figures: list of {description, label} for each figure/diagram
  3. Build separate chunks per content type so Pinecone retrieval is precise.

This approach works on scanned PDFs, mixed PDFs, and native PDFs alike.
"""

import base64
import json
import os
import time
from pathlib import Path

import openai
from pdf2image import convert_from_path

from config import IMAGE_DIR, VISION_MODEL, get_openai_api_key, get_openai_base_url

# ---------------------------------------------------------------------------
# Page → image conversion
# ---------------------------------------------------------------------------

def pdf_pages_to_images(pdf_path: str, dpi: int = 150) -> list[tuple[int, str]]:
    """
    Convert each PDF page to a PNG saved in IMAGE_DIR.
    Returns list of (page_number, image_path) tuples.
    """
    os.makedirs(IMAGE_DIR, exist_ok=True)
    stem = Path(pdf_path).stem
    images = convert_from_path(pdf_path, dpi=dpi, fmt="png")
    paths = []
    for i, img in enumerate(images, start=1):
        img_path = os.path.join(IMAGE_DIR, f"{stem}_page{i}.png")
        img.save(img_path, "PNG")
        paths.append((i, img_path))
    print(f"  Rendered {len(paths)} pages to images")
    return paths


# ---------------------------------------------------------------------------
# GPT-4o vision: structured page analysis
# ---------------------------------------------------------------------------

_PAGE_ANALYSIS_PROMPT = """You are analyzing a page from a technical product manual.
Carefully examine this page and return a JSON object with exactly these fields:

{
  "page_type": "<one of: text, figure, table, mixed>",
  "section_title": "<the section or heading on this page, or empty string>",
  "text_content": "<all readable text on this page, verbatim, preserving numbered steps and bullet points>",
  "tables": [
    "<markdown table 1>",
    "<markdown table 2>"
  ],
  "figures": [
    {
      "label": "<figure label or number if visible, else empty>",
      "description": "<detailed description: all labeled parts, arrows, callouts, component names, what assembly step or troubleshooting scenario it shows>"
    }
  ]
}

Rules:
- tables: use markdown pipe format. Include ALL rows and headers. Empty list if no tables.
- figures: describe every diagram, exploded view, illustration, or photo. Empty list if none.
- text_content: include ALL text not already captured in tables or figure labels.
- Be precise with part names, numbers, and technical terms exactly as written.
- Return ONLY the JSON object, no other text."""


def analyze_page_with_vision(page_num: int, image_path: str, client: openai.OpenAI) -> dict:
    """
    Send a page image to GPT-4o and get back structured content.
    Returns parsed dict or a minimal fallback on failure.
    """
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": _PAGE_ANALYSIS_PROMPT},
                    ],
                }
            ],
            max_tokens=1500,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        return json.loads(raw)

    except Exception as e:
        print(f"  [warn] Page {page_num} analysis failed: {e}")
        return {
            "page_type": "text",
            "section_title": "",
            "text_content": f"[Page {page_num} — could not be processed]",
            "tables": [],
            "figures": [],
        }


# ---------------------------------------------------------------------------
# Build chunks from page analysis results
# ---------------------------------------------------------------------------

def build_chunks_from_page(
    page_num: int,
    image_path: str,
    analysis: dict,
    pdf_name: str,
) -> list[dict]:
    """
    Convert GPT-4o page analysis into chunk dicts ready for embedding.
    Produces up to 3 chunk types per page: text, table (one per table), image (one per figure).
    """
    chunks = []
    section = analysis.get("section_title", "")

    # --- Text chunk ---
    text = (analysis.get("text_content") or "").strip()
    if text and len(text) > 30:
        chunks.append(
            {
                "content": text,
                "element_type": "text",
                "page_number": page_num,
                "section_title": section,
                "table_markdown": "",
                "original_content": text,
                "image_path": image_path,   # page image for reference
                "image_caption": "",
                "source_pdf": pdf_name,
            }
        )

    # --- Table chunks (one per table) ---
    for table_md in analysis.get("tables", []):
        table_md = (table_md or "").strip()
        if table_md and len(table_md) > 10:
            chunks.append(
                {
                    "content": table_md,
                    "element_type": "table",
                    "page_number": page_num,
                    "section_title": section,
                    "table_markdown": table_md,
                    "original_content": table_md,
                    "image_path": image_path,
                    "image_caption": "",
                    "source_pdf": pdf_name,
                }
            )

    # --- Figure chunks (one per figure) ---
    for fig in analysis.get("figures", []):
        label = fig.get("label", "")
        desc = (fig.get("description") or "").strip()
        if not desc:
            continue
        caption = f"{label}: {desc}" if label else desc
        chunks.append(
            {
                "content": caption,
                "element_type": "image",
                "page_number": page_num,
                "section_title": section,
                "table_markdown": "",
                "original_content": "",
                "image_path": image_path,
                "image_caption": caption,
                "source_pdf": pdf_name,
            }
        )

    return chunks


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: str, pdf_name: str) -> list[dict]:
    """
    Full pipeline: PDF → page images → GPT-4o vision analysis → chunks.
    Returns all chunks (text + table + image) ready for embedding.
    """
    client = openai.OpenAI(api_key=get_openai_api_key(), base_url=get_openai_base_url())

    print("  Converting pages to images...")
    page_images = pdf_pages_to_images(pdf_path)

    all_chunks = []
    for page_num, image_path in page_images:
        print(f"  Analyzing page {page_num}/{len(page_images)}...")
        analysis = analyze_page_with_vision(page_num, image_path, client)

        page_type = analysis.get("page_type", "text")
        n_tables = len(analysis.get("tables", []))
        n_figs = len(analysis.get("figures", []))
        print(f"    → type={page_type}, tables={n_tables}, figures={n_figs}")

        chunks = build_chunks_from_page(page_num, image_path, analysis, pdf_name)
        all_chunks.extend(chunks)
        time.sleep(0.5)   # avoid rate limits

    return all_chunks
