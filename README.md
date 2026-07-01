# PP-DocLayoutV3 ONNX Runner

Minimal Rust runner for PaddlePaddle PP-DocLayoutV3 ONNX using `ort = 2.0.0-rc.12`.

## Model

Download the ONNX model and metadata from Hugging Face:

```bash
uv --cache-dir models/.uv-cache sync
uv --cache-dir models/.uv-cache run python scripts/download_model.py
```

The upstream `inference.yml` defines:

- input image resized to `800x800`
- RGB `f32` values scaled by `1/255`
- CHW layout, batch size 1
- 25 PP-DocLayoutV3 labels

## Usage

Run detection on every page of a PDF:

```bash
cargo run --release -- detect path/to/document.pdf models/inference.onnx
```

The `detect` command renders PDF pages with pdfium at 96 DPI, applies the PP-DocLayoutV3 preprocessing, and writes one PNG plus one JSON file per page under `output/`:

```text
output/page-0001.png
output/page-0001.json
```

## Browser WASM test page

Build the browser WASM package:

```bash
wasm-pack build --target web --out-dir web/pkg --no-default-features --features wasm
```

Start a local HTTP server from the project root:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Open the test page:

```text
http://127.0.0.1:8000/web/index.html
```

If port `8000` is already in use, choose another port, for example:

```bash
python3 -m http.server 8001 --bind 127.0.0.1
```
