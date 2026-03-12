import cv2
import numpy as np
import os
import json
import base64
import re
import time
import sys

# Path
PDF_PATHS = [
    "input3.pdf",   
]
OUTPUT_DIR = "catalog_output"
IMG_DIR    = os.path.join(OUTPUT_DIR, "product_images")
RENDER_DPI = 200
ANALYSIS_MAX_W = 1200
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

os.makedirs(IMG_DIR, exist_ok=True)



def render_pdf_pages(pdf_path, dpi=200):
    """
    Render all PDF pages to numpy arrays.
    Tries pymupdf first (pip install pymupdf), then pdf2image (pip install pdf2image).
    """
    
    try:
        import fitz  
        doc = fitz.open(pdf_path)
        pages = []
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            pages.append(img)
        print(f"  Rendered {len(pages)} pages via PyMuPDF at {dpi} DPI")
        return pages
    except ImportError:
        pass
    except Exception as e:
        print(f"  PyMuPDF error: {e}")

    
    try:
        from pdf2image import convert_from_path 
        pil_pages = convert_from_path(pdf_path, dpi=dpi)
        pages = []
        for pil_img in pil_pages:
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            pages.append(img)
        print(f"  Rendered {len(pages)} pages via pdf2image at {dpi} DPI")
        return pages
    except ImportError:
        pass
    except Exception as e:
        print(f"  pdf2image error: {e}")

    print("ERROR: No PDF renderer found. Install one of:")
    print("  pip install pymupdf")
    print("  pip install pdf2image   (also needs poppler on PATH)")
    sys.exit(1)




def encode_image_b64(img_array):
    _, buf = cv2.imencode(".jpg", img_array, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode()


def call_gpt4o(image_b64, prompt, max_tokens=2048):
    import urllib.request
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = json.dumps({
        "model": "gpt-4o",
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"}},
                {"type": "text", "text": prompt},
            ],
        }],
    }).encode()

    for attempt in range(4):
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body, headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            status = e.code
            err_body = e.read().decode()[:300]
            print(f"    API HTTP {status}: {err_body}")
            if status == 429:
                wait = 10 * (attempt + 1)
                print(f"    Rate limited - waiting {wait}s ...")
                time.sleep(wait)
            else:
                break
        except Exception as ex:
            print(f"    API error: {ex}")
            break
    return ""


def downscale_for_analysis(img, max_w=ANALYSIS_MAX_W):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    return cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)


PAGE_PROMPT = """\
This is a page from a clothing catalog. Analyze it carefully.

TASK: Return a single JSON object (no markdown, no extra text) with this exact structure:

{
  "page_type": "multi_product",
  "common_info": {
    "brand": "",
    "style": "",
    "fabric": "",
    "sizes": "",
    "pack": "",
    "mrp": "",
    "product_code": "",
    "design_no": "",
    "collection": ""
  },
  "products": [
    {
      "color": "#COLOR or color name",
      "left": 0,
      "top": 0,
      "width": 100,
      "height": 100
    }
  ]
}

RULES:
- page_type must be one of: "multi_product", "single_product", "intro"
- "intro": brand intro, divider, or no purchasable products -> products: []
- "single_product": one style on the page (may show 2 model photos). Return ONE product entry covering the model photo area.
- "multi_product": grid of color variants. Return ONE entry PER color cell.
- Bounding boxes are PERCENTAGES of page dimensions (0-100), not pixels.
- left/top = top-left corner of product image cell. width/height = size of that cell.
- Do NOT include the bottom info banner, page borders, or section headers in any bounding box.
- common_info applies to ALL products on the page (read from the banner/caption text).
- For color: read the color swatch label or text. Use "" if unclear.
- Use "" for any common_info field you cannot read.
"""


def crop_product(full_img, bbox, padding_pct=0.005):
    h, w = full_img.shape[:2]
    pad_x = int(w * padding_pct)
    pad_y = int(h * padding_pct)

    x1 = max(0, int(w * bbox["left"]   / 100) - pad_x)
    y1 = max(0, int(h * bbox["top"]    / 100) - pad_y)
    x2 = min(w, int(w * (bbox["left"] + bbox["width"])  / 100) + pad_x)
    y2 = min(h, int(h * (bbox["top"]  + bbox["height"]) / 100) + pad_y)

    cw, ch = x2 - x1, y2 - y1
    if cw < 50 or ch < 50:
        return None
    if (cw * ch) / (w * h) > 0.88:
        return None
    return full_img[y1:y2, x1:x2]


def process_page(full_img, page_num, product_id_start, pdf_name):
    analysis_img = downscale_for_analysis(full_img)
    b64 = encode_image_b64(analysis_img)

    print(f"  [p{page_num}] Sending to GPT-4o ({analysis_img.shape[1]}x{analysis_img.shape[0]}) ...", end=" ", flush=True)
    raw = call_gpt4o(b64, PAGE_PROMPT, max_tokens=2048)

    if not raw:
        print("NO RESPONSE")
        return []

    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        print(f"JSON error: {e}\nRaw: {raw[:300]}")
        return []

    page_type = data.get("page_type", "intro")
    print(f"type={page_type}", end=" ")

    if page_type == "intro" or not data.get("products"):
        print("-> skipped")
        return []

    common = data.get("common_info", {})
    products_out = []
    pid = product_id_start

    for prod in data["products"]:
        crop = crop_product(full_img, prod)
        if crop is None:
            print("x", end="", flush=True)
            continue

        img_filename = f"product_{pid:04d}.jpg"
        img_path = os.path.join(IMG_DIR, img_filename)
        cv2.imwrite(img_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

        entry = {
            "product_id": pid,
            "page": page_num,
            "pdf": pdf_name,
            "brand":        common.get("brand", ""),
            "style":        common.get("style", ""),
            "color":        prod.get("color", ""),
            "fabric":       common.get("fabric", ""),
            "sizes":        common.get("sizes", ""),
            "pack":         common.get("pack", ""),
            "mrp":          common.get("mrp", ""),
            "product_code": common.get("product_code", ""),
            "design_no":    common.get("design_no", ""),
            "collection":   common.get("collection", ""),
            "image":        f"product_images/{img_filename}",
            "bbox": {
                "left":   prod.get("left", 0),
                "top":    prod.get("top", 0),
                "width":  prod.get("width", 100),
                "height": prod.get("height", 100),
            },
        }
        products_out.append(entry)
        pid += 1

    print(f"-> {len(products_out)} products")
    return products_out


def run():
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set in script.")
        sys.exit(1)

    start_page = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    end_page   = int(sys.argv[2]) if len(sys.argv) > 2 else 9999

    all_products = []
    product_id = 1

    for pdf_path in PDF_PATHS:
        pdf_name = os.path.basename(pdf_path)
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_name}")
        print(f"{'='*60}")

        pages = render_pdf_pages(pdf_path, dpi=RENDER_DPI)
        if not pages:
            print("  No pages rendered - skipping.")
            continue

        for i, full_img in enumerate(pages):
            page_num = i + 1
            if page_num < start_page or page_num > end_page:
                continue

            prods = process_page(full_img, page_num, product_id, pdf_name)
            all_products.extend(prods)
            product_id += len(prods)

    json_path = os.path.join(OUTPUT_DIR, "products.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_products, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Done. Extracted {len(all_products)} products.")
    print(f"JSON saved -> {json_path}")
    print(f"Images in -> {IMG_DIR}")


if __name__ == "__main__":
    run()