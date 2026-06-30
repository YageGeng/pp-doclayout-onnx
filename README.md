# PP-DocLayoutV3 ONNX Runner

Minimal Rust runner for PaddlePaddle PP-DocLayoutV3 ONNX using `ort = 2.0.0-rc.12`.

## Model

Download the ONNX model from Hugging Face:

```bash
mkdir -p models
curl -L https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx/resolve/main/inference.onnx -o models/inference.onnx
```

The upstream `inference.yml` defines:

- input image resized to `800x800`
- RGB `f32` values scaled by `1/255`
- CHW layout, batch size 1
- 25 PP-DocLayoutV3 labels

## Usage

Inspect model inputs and outputs:

```bash
cargo run --release -- inspect --model models/inference.onnx
```

Run detection on every page of a PDF:

```bash
cargo run --release -- detect path/to/document.pdf models/inference.onnx
```

The `detect` command renders PDF pages with pdfium at 96 DPI, applies the PP-DocLayoutV3 preprocessing, and writes one PNG plus one JSON file per page under `output/`:

```text
output/page-0001.png
output/page-0001.json
```
