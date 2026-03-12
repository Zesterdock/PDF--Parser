import os
import cv2
import fitz
import json
import base64
import numpy as np
import urllib.request
import re
import time



PDF_PATH = "input2.pdf"
OUTPUT_DIR = "catalog_output"
IMG_DIR = os.path.join(OUTPUT_DIR, "product_images")

DPI = 300
MIN_PRODUCT_AREA = 180000

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

os.makedirs(IMG_DIR, exist_ok=True)



def render_pdf(pdf_path):

    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(DPI/72, DPI/72)

    pages = []

    for page in doc:

        pix = page.get_pixmap(matrix=mat)

        img = np.frombuffer(
            pix.samples,
            dtype=np.uint8
        ).reshape(pix.height, pix.width, pix.n)

        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        pages.append(img)

    print("Rendered", len(pages), "pages")

    return pages




def split_spread(page):

    h, w = page.shape[:2]

    left  = page[:, :w//2]
    right = page[:, w//2:]

    return [left, right]



def detect_products(page):

    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 40, 120)

    kernel = np.ones((9,9), np.uint8)

    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours,_ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for c in contours:

        x,y,w,h = cv2.boundingRect(c)

        area = w*h

        if area < MIN_PRODUCT_AREA:
            continue

        aspect = w/h

        if aspect < 0.25 or aspect > 4:
            continue

        boxes.append((x,y,w,h))

    return boxes




def merge_boxes(boxes):

    merged = []

    for box in boxes:

        x,y,w,h = box
        added = False

        for i,(mx,my,mw,mh) in enumerate(merged):

            if abs(x-mx) < 100 and abs(y-my) < 100:

                nx = min(x,mx)
                ny = min(y,my)

                nw = max(x+w,mx+mw) - nx
                nh = max(y+h,my+mh) - ny

                merged[i] = (nx,ny,nw,nh)
                added = True
                break

        if not added:
            merged.append(box)

    return merged



def encode_image(img):

    _,buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return base64.b64encode(buf).decode()


def call_gpt(image_b64):

    prompt = """
This image shows a product panel from an electronics catalog.

Extract product info and return ONLY JSON:

{
"name":"",
"model_no":"",
"category":"",
"key_features":[],
"mrp":""
}
"""

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    body = json.dumps({

        "model":"gpt-4o",

        "messages":[{

            "role":"user",

            "content":[
                {
                    "type":"image_url",
                    "image_url":{
                        "url":f"data:image/jpeg;base64,{image_b64}"
                    }
                },
                {
                    "type":"text",
                    "text":prompt
                }
            ]
        }]

    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers=headers,
        method="POST"
    )

    try:

        with urllib.request.urlopen(req, timeout=90) as resp:

            text = json.loads(resp.read())["choices"][0]["message"]["content"]

            text = re.sub(r"```json|```","",text)

            return json.loads(text)

    except:
        return None



def run():

    pages = render_pdf(PDF_PATH)

    products = []
    pid = 1

    for spread_i,spread in enumerate(pages):

        print("Processing spread",spread_i+1)

        pages_split = split_spread(spread)

        for page in pages_split:

            boxes = detect_products(page)

            boxes = merge_boxes(boxes)

            for box in boxes:

                x,y,w,h = box

                crop = page[y:y+h, x:x+w]

                if crop.shape[0] < 250:
                    continue

                fname = f"product_{pid:04d}.jpg"

                img_path = os.path.join(IMG_DIR, fname)

                cv2.imwrite(img_path, crop)

                print("Saved", fname)

                meta = None

                if OPENAI_API_KEY:

                    meta = call_gpt(encode_image(crop))

                    time.sleep(1)

                products.append({

                    "product_id":pid,
                    "image":f"product_images/{fname}",
                    "metadata":meta
                })

                pid += 1

    with open(os.path.join(OUTPUT_DIR,"products.json"),"w") as f:

        json.dump(products,f,indent=2)

    print("\nDone")
    print("Products:",len(products))


if __name__ == "__main__":
    run()