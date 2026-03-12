# PDF Product Catalog Parser

Automated extraction of product information from PDF catalogs using GPT-4 Vision and computer vision techniques.

## Features

- 🤖 **AI-Powered Detection**: Uses GPT-4 Vision to intelligently identify products on catalog pages
- 📦 **Multi-Format Support**: Handles both single-page and spread (two-page) PDF layouts
- 🎯 **Smart Cropping**: Automatically extracts individual product images with accurate boundaries
- 📊 **Structured Output**: Generates clean JSON data with product details, images, and metadata
- 🔧 **Flexible**: Works with various catalog formats (clothing, electronics, etc.)

## Scripts

### 1. `catalog.py` - Clothing Catalog Extractor
Optimized for clothing/apparel catalogs with color variants.

**Best for:**
- T-shirts, shirts, clothing items
- Multiple color variants per page
- Common product info (fabric, style, size) at bottom of pages

### 2. `bpl.py` - Electronics Catalog Extractor  
Optimized for electronics catalogs with spread layouts.

**Best for:**
- Headphones, speakers, audio equipment
- PDF spreads (two pages side-by-side)
- Products with detailed specifications

## Installation

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

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY=your-api-key-here
```

Get your API key from: https://platform.openai.com/api-keys

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

### BPL.py (Electronics Catalogs)

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

## Troubleshooting

### "ERROR: OPENAI_API_KEY not set"
Make sure you've set the environment variable correctly. Try running:
```bash
echo %OPENAI_API_KEY%    # Windows CMD
echo $env:OPENAI_API_KEY # Windows PowerShell
echo $OPENAI_API_KEY     # Linux/Mac
```

### "No PDF renderer found"
Install PyMuPDF:
```bash
pip install pymupdf
```

### Rate limiting (HTTP 429)
The script automatically retries with exponential backoff. If you hit rate limits frequently:
- Reduce `max_tokens` in the API call
- Process fewer pages at once
- Upgrade your OpenAI API plan

### Poor extraction quality
- Increase `RENDER_DPI` for better image quality (e.g., 300)
- Check that your PDF contains actual images, not just scanned pages
- Adjust bounding box size filters in the code if products are being skipped

## Cost Estimation

Using GPT-4 Vision (gpt-4o):
- ~$0.01-0.03 per catalog page
- Depends on image size and max_tokens setting
- 100-page catalog: approximately $1-3

## Requirements

- Python 3.8+
- OpenAI API key with GPT-4 Vision access
- Windows/Linux/Mac

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
