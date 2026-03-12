import cv2
import numpy as np
import os
import json
import base64
import re
import time
import sys
import urllib.request

# ============================================================================
# CONFIGURATION PRESETS
# ============================================================================

CATALOG_CONFIGS = {
    "clothing": {
        "name": "Clothing Catalog (Multi-color variants)",
        "dpi": 200,
        "analysis_max_w": 1200,
        "use_ai_detection": True,
        "split_spreads": False,
        "prompt": """This is a page from a clothing catalog. Analyze it carefully.

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
    },
    
    "electronics": {
        "name": "Electronics Catalog (BPL Audio)",
        "dpi": 300,
        "analysis_max_w": 1400,
        "use_ai_detection": True,
        "split_spreads": True,
        "prompt": """This is a page from an electronics catalog showing audio equipment.

TASK: Return a single JSON object (no markdown, no extra text) with this exact structure:

{
  "page_type": "multi_product",
  "products": [
    {
      "name": "",
      "model_no": "",
      "category": "",
      "key_features": [],
      "mrp": "",
      "left": 0,
      "top": 0,
      "width": 100,
      "height": 100
    }
  ]
}

RULES:
- page_type must be one of: "multi_product", "single_product", "intro"
- "intro": cover page, section divider, or no specific products -> products: []
- Return ONE entry PER product shown on the page.
- Bounding boxes are PERCENTAGES of page dimensions (0-100), not pixels.
- left/top = top-left corner of product image. width/height = size of product region.
- Extract model numbers (e.g., "T54NS22A", "BPL 101C") precisely.
- category examples: "Soundbar", "Home Theater", "Speaker", "Subwoofer"
- key_features: array of 3-5 main features (e.g., ["6.5 inch woofer", "Bluetooth 5.0", "240W output"])
- Use "" for fields you cannot read clearly.
"""
    }
}

# ============================================================================
# RUNTIME CONFIGURATION (modify this or pass via command line)
# ============================================================================

# Select config: "clothing" or "electronics"
ACTIVE_CONFIG = "clothing"

# Input/Output paths
PDF_PATHS = ["input3.pdf"]
OUTPUT_DIR = "catalog_output1"
IMG_DIR = os.path.join(OUTPUT_DIR, "product_images")

# API Key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

os.makedirs(IMG_DIR, exist_ok=True)

# ============================================================================
# PDF RENDERING
# ============================================================================

def render_pdf_pages(pdf_path, dpi=200):
    """Render all PDF pages to numpy arrays using PyMuPDF."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        pages = []
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        for page in doc:
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
        print("ERROR: PyMuPDF not found. Install: pip install pymupdf")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR rendering PDF: {e}")
        sys.exit(1)

# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def split_spread(page_img):
    """Split landscape page into left and right halves."""
    h, w = page_img.shape[:2]
    left = page_img[:, :w//2]
    right = page_img[:, w//2:]
    return [left, right]

def downscale_for_analysis(img, max_w=1200):
    """Downscale image for API efficiency while maintaining aspect ratio."""
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    return cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)

def crop_product(full_img, bbox, padding_pct=0.005):
    """Crop product from full page image using percentage-based bounding box."""
    h, w = full_img.shape[:2]
    pad_x = int(w * padding_pct)
    pad_y = int(h * padding_pct)

    x1 = max(0, int(w * bbox["left"] / 100) - pad_x)
    y1 = max(0, int(h * bbox["top"] / 100) - pad_y)
    x2 = min(w, int(w * (bbox["left"] + bbox["width"]) / 100) + pad_x)
    y2 = min(h, int(h * (bbox["top"] + bbox["height"]) / 100) + pad_y)

    cw, ch = x2 - x1, y2 - y1
    if cw < 50 or ch < 50:
        return None
    if (cw * ch) / (w * h) > 0.88:  # Skip if crop is almost entire page
        return None
    return full_img[y1:y2, x1:x2]

# ============================================================================
# GPT-4 VISION API
# ============================================================================

def encode_image_b64(img_array):
    """Encode numpy image array to base64 JPEG."""
    _, buf = cv2.imencode(".jpg", img_array, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode()

def call_gpt4o(image_b64, prompt, max_tokens=2048):
    """Call GPT-4o Vision API with retry logic."""
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
            print(f"\n    API HTTP {status}: {err_body}")
            if status == 429:
                wait = 10 * (attempt + 1)
                print(f"    Rate limited - waiting {wait}s ...")
                time.sleep(wait)
            else:
                break
        except Exception as ex:
            print(f"\n    API error: {ex}")
            break
    return ""

# ============================================================================
# PRODUCT EXTRACTION
# ============================================================================

def process_page_clothing(full_img, page_num, product_id_start, pdf_name, config):
    """Extract products from clothing catalog page."""
    analysis_img = downscale_for_analysis(full_img, config["analysis_max_w"])
    b64 = encode_image_b64(analysis_img)

    print(f"  [p{page_num}] Analyzing ({analysis_img.shape[1]}x{analysis_img.shape[0]}) ...", end=" ", flush=True)
    raw = call_gpt4o(b64, config["prompt"], max_tokens=2048)

    if not raw:
        print("NO RESPONSE")
        return []

    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        print(f"JSON error: {e}")
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
            "brand": common.get("brand", ""),
            "style": common.get("style", ""),
            "color": prod.get("color", ""),
            "fabric": common.get("fabric", ""),
            "sizes": common.get("sizes", ""),
            "pack": common.get("pack", ""),
            "mrp": common.get("mrp", ""),
            "product_code": common.get("product_code", ""),
            "design_no": common.get("design_no", ""),
            "collection": common.get("collection", ""),
            "image": f"product_images/{img_filename}",
            "bbox": prod,
        }
        products_out.append(entry)
        pid += 1

    print(f"-> {len(products_out)} products")
    return products_out

def process_page_electronics(full_img, page_num, product_id_start, pdf_name, config):
    """Extract products from electronics catalog page."""
    analysis_img = downscale_for_analysis(full_img, config["analysis_max_w"])
    b64 = encode_image_b64(analysis_img)

    print(f"  [p{page_num}] Analyzing ({analysis_img.shape[1]}x{analysis_img.shape[0]}) ...", end=" ", flush=True)
    raw = call_gpt4o(b64, config["prompt"], max_tokens=2048)

    if not raw:
        print("NO RESPONSE")
        return []

    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        print(f"JSON error: {e}")
        return []

    page_type = data.get("page_type", "intro")
    print(f"type={page_type}", end=" ")

    if page_type == "intro" or not data.get("products"):
        print("-> skipped")
        return []

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
            "name": prod.get("name", ""),
            "model_no": prod.get("model_no", ""),
            "category": prod.get("category", ""),
            "key_features": prod.get("key_features", []),
            "mrp": prod.get("mrp", ""),
            "image": f"product_images/{img_filename}",
            "bbox": {
                "left": prod.get("left", 0),
                "top": prod.get("top", 0),
                "width": prod.get("width", 100),
                "height": prod.get("height", 100),
            },
        }
        products_out.append(entry)
        pid += 1

    print(f"-> {len(products_out)} products")
    return products_out

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run():
    """Main extraction workflow."""
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    # Load configuration
    if ACTIVE_CONFIG not in CATALOG_CONFIGS:
        print(f"ERROR: Invalid ACTIVE_CONFIG '{ACTIVE_CONFIG}'. Choose from: {list(CATALOG_CONFIGS.keys())}")
        sys.exit(1)
    
    config = CATALOG_CONFIGS[ACTIVE_CONFIG]
    print(f"\n{'='*60}")
    print(f"Configuration: {config['name']}")
    print(f"DPI: {config['dpi']}, Split Spreads: {config['split_spreads']}")
    print(f"{'='*60}\n")

    # Parse optional page range from command line
    start_page = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    end_page = int(sys.argv[2]) if len(sys.argv) > 2 else 9999

    all_products = []
    product_id = 1

    # Select processing function based on config
    if ACTIVE_CONFIG == "clothing":
        process_func = process_page_clothing
    elif ACTIVE_CONFIG == "electronics":
        process_func = process_page_electronics
    else:
        process_func = process_page_clothing  # default

    for pdf_path in PDF_PATHS:
        pdf_name = os.path.basename(pdf_path)
        print(f"\nProcessing: {pdf_name}")
        print("-" * 60)

        pages = render_pdf_pages(pdf_path, dpi=config["dpi"])
        if not pages:
            print("  No pages rendered - skipping.")
            continue

        for i, page_img in enumerate(pages):
            page_num = i + 1
            if page_num < start_page or page_num > end_page:
                continue

            # Handle spread splitting if enabled
            if config["split_spreads"] and page_img.shape[1] > page_img.shape[0] * 1.3:
                print(f"\n  [p{page_num}] Detected spread - splitting into left/right")
                sub_pages = split_spread(page_img)
                for sub_idx, sub_page in enumerate(sub_pages):
                    sub_label = f"{page_num}.{sub_idx+1}"
                    prods = process_func(sub_page, sub_label, product_id, pdf_name, config)
                    all_products.extend(prods)
                    product_id += len(prods)
            else:
                prods = process_func(page_img, page_num, product_id, pdf_name, config)
                all_products.extend(prods)
                product_id += len(prods)

    # Save results
    json_path = os.path.join(OUTPUT_DIR, "products.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_products, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✓ Extraction complete!")
    print(f"  Total products: {len(all_products)}")
    print(f"  JSON: {json_path}")
    print(f"  Images: {IMG_DIR}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run()
