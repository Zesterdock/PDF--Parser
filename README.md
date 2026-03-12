# PDF Product Catalog Parser

## Scripts

### 1. `catalog.py` - Clothing Catalog Extractor
Optimized for clothing/apparel catalogs with color variants.

**Best for:**
- T-shirts, shirts, clothing items
- Multiple color variants per page
- Common product info (fabric, style, size) at bottom of pages

### 2. `bpl.py` - Electronics Catalog Extractor  
Optimized for electronics catalogs with spread layouts.
 ###Installation
### 1. Clone the repository
```bash
git clone https://github.com/Zesterdock/PDF--Parser.git
cd PDF--Parser
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up OpenAI API Key

**Windows (CMD):**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

## Usage

### Catalog.py (Clothing Catalogs)

1. **Configure the script:**
   Edit `catalog.py` and set your PDF path:
   ```python
   PDF_PATHS = [
       "your_catalog.pdf",   
   ]
   ```

2. **Run the script:**
   ```bash
   # Process all pages
   python catalog.py
   
   # Process specific page range (e.g., pages 5-10)
   python catalog.py 5 10
   ```

3. **Output:**
   - `catalog_output/products.json` - Product data
   - `catalog_output/product_images/` - Cropped product images

### BPL.py 

1. **Configure the script:**
   Edit `bpl.py` and set your PDF path:
   ```python
   PDF_PATHS = [
       "electronics_catalog.pdf",
   ]
   ```

2. **Run the script:**
   ```bash
   # Process all pages
   python bpl.py
   
   # Process specific page range
   python bpl.py 1 20
   ```

3. **Output:**
   - `bpl_catalog_output/products.json` - Product data
   - `bpl_catalog_output/product_images/` - Cropped product images

## Output Format

### catalog.py JSON Structure
```json
{
  "product_id": 1,
  "page": 4,
  "pdf": "input.pdf",
  "brand": "",
  "style": "2076",
  "color": "#SOFT BEIGE",
  "fabric": "Imported Canvas",
  "sizes": "M/L/XL/2XL",
  "pack": "8",
  "mrp": "₹559/-",
  "product_code": "",
  "design_no": "",
  "collection": "",
  "image": "product_images/product_0001.jpg",
  "bbox": {
    "left": 5.2,
    "top": 12.8,
    "width": 18.4,
    "height": 42.1
  }
}
```

### bpl.py JSON Structure
```json
{
  "product_id": 1,
  "page": "spread2L",
  "pdf": "catalog.pdf",
  "name": "BPL Wireless Headphones",
  "model_no": "BPL-H100",
  "category": "Headphones",
  "color": "Black",
  "key_features": [
    "40mm drivers",
    "20 hour battery",
    "Bluetooth 5.0"
  ],
  "mrp": "₹1999/-",
  "image": "product_images/product_0001.jpg",
  "bbox": {...}
}
```

## Configuration

### catalog.py Settings
```python
RENDER_DPI = 200           # PDF rendering quality
ANALYSIS_MAX_W = 1200      # Max width for AI analysis
OUTPUT_DIR = "catalog_output"
```

### bpl.py Settings
```python
RENDER_DPI = 200           # PDF rendering quality
ANALYSIS_MAX_W = 1400      # Max width for AI analysis (wider for spreads)
SPLIT_SPREADS = True       # Split two-page spreads
OUTPUT_DIR = "bpl_catalog_output"
```


## Dependencies

See `requirements.txt`:
- opencv-python - Image processing
- numpy - Array operations
- pymupdf - PDF rendering
- openai - GPT-4 Vision API

## License

MIT License - See LICENSE file

## Contributing

Pull requests welcome! Please open an issue first to discuss changes.

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing issues for solutions

## Acknowledgments

- OpenAI GPT-4 Vision API
- PyMuPDF (fitz) for PDF handling
- OpenCV for image processing

