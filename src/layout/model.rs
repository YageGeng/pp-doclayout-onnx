use std::{fs, path::Path};

use anyhow::{anyhow, bail, Result};
use image::DynamicImage;
use ort::{session::Session, value::TensorRef};
use serde::Serialize;

use super::{
    annotate_page_rgba, parse_detr_outputs, parse_paddle_fetch_output, Detection,
    PdfPageDetections, PdfPageOutputFile, PreprocessedImage, DEFAULT_DPI, DEFAULT_OUTPUT_DIR,
    DEFAULT_THRESHOLD,
};
use crate::pdf::{encode_png_rgba, output_path_for_page, PdfiumSession};
use crate::PPDocLayoutV3Label;

/// Raw f32 output tensor sample used when debugging exporter output formats.
#[derive(Debug, Clone, Serialize)]
pub struct RawTensorDump {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

/// ONNX Runtime-backed PP-DocLayout detector that owns a loaded model session.
pub struct OrtDocLayout {
    session: Session,
}

impl OrtDocLayout {
    /// Creates an ONNX Runtime-backed PP-DocLayout detector from a model file.
    pub fn new(model_path: impl AsRef<Path>, intra_threads: Option<usize>) -> Result<Self> {
        let model_path = model_path.as_ref();
        ensure_model_exists(model_path)?;

        let mut builder = Session::builder()
            .map_err(|error| anyhow!("create ONNX Runtime session builder: {error}"))?;
        if let Some(threads) = intra_threads {
            builder = builder
                .with_intra_threads(threads)
                .map_err(|error| anyhow!("configure intra-op thread count: {error}"))?;
        }

        let session = builder
            .commit_from_file(model_path)
            .map_err(|error| anyhow!("load ONNX model {}: {error}", model_path.display()))?;
        Ok(Self { session })
    }

    /// Runs layout detection on a single image file.
    pub fn detect_image_path(
        &mut self,
        image_path: impl AsRef<Path>,
        threshold: f32,
    ) -> Result<Vec<Detection>> {
        let image_path = image_path.as_ref();
        let image = image::open(image_path)
            .map_err(|error| anyhow!("open input image {}: {error}", image_path.display()))?;
        let input = PreprocessedImage::try_from(&image)?;
        self.detect_preprocessed(input, threshold)
    }

    /// Runs layout detection on every rendered page of a PDF.
    pub fn detect_pdf_path(
        &mut self,
        pdf_path: impl AsRef<Path>,
    ) -> Result<Vec<PdfPageDetections>> {
        let pdfium = PdfiumSession::new();
        let pdf = pdfium.open_document(pdf_path)?;
        let mut pages = Vec::new();
        pdf.visit_rendered_pages(DEFAULT_DPI, |rendered| {
            let rgb = DynamicImage::ImageRgba8(rendered.rgba.clone()).to_rgb8();
            let input = PreprocessedImage::try_from(&rgb)?;
            let detections = self.detect_preprocessed(input, DEFAULT_THRESHOLD)?;
            pages.push(PdfPageDetections {
                page_number: rendered.page_number,
                width: rendered.width,
                height: rendered.height,
                page_width: rendered.page_width,
                page_height: rendered.page_height,
                dpi: DEFAULT_DPI,
                detections,
            });
            Ok(())
        })?;

        Ok(pages)
    }

    /// Runs an image through the model and returns raw output tensor samples.
    pub fn dump_image_outputs(
        &mut self,
        image_path: impl AsRef<Path>,
        max_values: usize,
    ) -> Result<Vec<RawTensorDump>> {
        let image_path = image_path.as_ref();
        let image = image::open(image_path)
            .map_err(|error| anyhow!("open input image {}: {error}", image_path.display()))?;
        let input = PreprocessedImage::try_from(&image)?;
        let outputs = self.run_preprocessed(input)?;

        let mut dumps = Vec::new();
        for (name, value) in outputs.iter() {
            let Ok(array) = value.try_extract_array::<f32>() else {
                continue;
            };
            dumps.push(RawTensorDump {
                name: name.to_string(),
                shape: array.shape().to_vec(),
                values: array.iter().copied().take(max_values).collect(),
            });
        }
        Ok(dumps)
    }

    /// Runs model inference on already preprocessed tensors and parses detections.
    pub fn detect_preprocessed(
        &mut self,
        input: PreprocessedImage,
        threshold: f32,
    ) -> Result<Vec<Detection>> {
        let original_size = input.original_size;
        let outputs = self.run_preprocessed(input)?;

        let mut arrays = Vec::new();
        let mut paddle_fetch_output = None;
        for (name, value) in outputs.iter() {
            let Ok(array) = value.try_extract_array::<f32>() else {
                continue;
            };
            let shape = array.shape().to_vec();
            if shape.len() == 2 && matches!(shape.get(1).copied(), Some(6 | 7)) {
                paddle_fetch_output = Some(array.to_owned().into_dyn());
            }
            arrays.push((name.to_string(), shape, array.to_owned()));
        }

        if let Some(output) = paddle_fetch_output {
            return parse_paddle_fetch_output(output, original_size, threshold);
        }

        let (_, _, logits) = arrays
            .iter()
            .find(|(_, shape, _)| {
                shape.len() == 3 && shape[0] == 1 && shape[2] >= PPDocLayoutV3Label::class_count()
            })
            .ok_or_else(|| {
                anyhow!(
                    "could not find DETR logits output; available outputs: {}",
                    format_output_shapes(&arrays)
                )
            })?;
        let (_, _, boxes) = arrays
            .iter()
            .find(|(_, shape, _)| shape.len() == 3 && shape[0] == 1 && shape[2] == 4)
            .ok_or_else(|| {
                anyhow!(
                    "could not find DETR boxes output; available outputs: {}",
                    format_output_shapes(&arrays)
                )
            })?;

        parse_detr_outputs(
            logits.clone().into_dyn(),
            boxes.clone().into_dyn(),
            original_size,
            threshold,
        )
    }

    /// Sends preprocessed tensors to ONNX Runtime and returns raw session outputs.
    fn run_preprocessed(
        &mut self,
        input: PreprocessedImage,
    ) -> Result<ort::session::SessionOutputs<'_>> {
        let image_tensor = TensorRef::from_array_view(&input.tensor)
            .map_err(|error| anyhow!("create ONNX tensor from preprocessed image: {error}"))?;
        let im_shape_tensor = TensorRef::from_array_view(&input.im_shape)
            .map_err(|error| anyhow!("create ONNX im_shape tensor: {error}"))?;
        let scale_factor_tensor = TensorRef::from_array_view(&input.scale_factor)
            .map_err(|error| anyhow!("create ONNX scale_factor tensor: {error}"))?;
        self.session
            .run(ort::inputs! {
                "im_shape" => im_shape_tensor,
                "image" => image_tensor,
                "scale_factor" => scale_factor_tensor,
            })
            .map_err(|error| anyhow!("run ONNX inference: {error}"))
    }

    /// Detects layouts in a PDF and writes annotated PNG plus JSON per page.
    pub fn detect_pdf_to_output_dir(
        &mut self,
        pdf_path: impl AsRef<Path>,
        output_dir: impl AsRef<Path>,
    ) -> Result<Vec<PdfPageOutputFile>> {
        let output_dir = output_dir.as_ref();
        fs::create_dir_all(output_dir)
            .map_err(|error| anyhow!("create output dir {}: {error}", output_dir.display()))?;

        let pdfium = PdfiumSession::new();
        let pdf = pdfium.open_document(pdf_path)?;
        let mut outputs = Vec::new();
        pdf.visit_rendered_pages(DEFAULT_DPI, |mut rendered| {
            let rgb = DynamicImage::ImageRgba8(rendered.rgba.clone()).to_rgb8();
            let input = PreprocessedImage::try_from(&rgb)?;
            let detections = self.detect_preprocessed(input, DEFAULT_THRESHOLD)?;
            annotate_page_rgba(&mut rendered.rgba, &detections);

            let image_path = output_path_for_page(output_dir, rendered.page_number, "png");
            let json_path = output_path_for_page(output_dir, rendered.page_number, "json");
            let png_bytes = encode_png_rgba(&rendered.rgba)
                .map_err(|error| anyhow!("encode page {} PNG: {error}", rendered.page_number))?;
            fs::write(&image_path, png_bytes)
                .map_err(|error| anyhow!("write {}: {error}", image_path.display()))?;

            let page_output = PdfPageDetections {
                page_number: rendered.page_number,
                width: rendered.width,
                height: rendered.height,
                page_width: rendered.page_width,
                page_height: rendered.page_height,
                dpi: DEFAULT_DPI,
                detections,
            };
            let json = serde_json::to_vec_pretty(&page_output).map_err(|error| {
                anyhow!("serialize page {} JSON: {error}", rendered.page_number)
            })?;
            fs::write(&json_path, json)
                .map_err(|error| anyhow!("write {}: {error}", json_path.display()))?;

            outputs.push(PdfPageOutputFile {
                page_number: rendered.page_number,
                image_path,
                json_path,
                detections: page_output.detections.len(),
            });
            Ok(())
        })?;

        Ok(outputs)
    }
}

/// Detects layouts in a PDF using default output and detection settings.
pub fn detect_pdf_to_output_dir(
    pdf_path: impl AsRef<Path>,
    model_path: impl AsRef<Path>,
) -> Result<Vec<PdfPageOutputFile>> {
    let mut detector = OrtDocLayout::new(model_path, None)?;
    detector.detect_pdf_to_output_dir(pdf_path, DEFAULT_OUTPUT_DIR)
}

/// Verifies the model file exists before ONNX Runtime tries to open it.
fn ensure_model_exists(model_path: &Path) -> Result<()> {
    if model_path.exists() {
        return Ok(());
    }

    bail!(
        "model file does not exist: {}\ndownload it with:\n  uv --cache-dir models/.uv-cache run python scripts/download_model.py",
        model_path.display()
    );
}

/// Formats model output tensor names and shapes for parser error messages.
fn format_output_shapes(outputs: &[(String, Vec<usize>, ndarray::ArrayD<f32>)]) -> String {
    outputs
        .iter()
        .map(|(name, shape, _)| format!("{name}: {shape:?}"))
        .collect::<Vec<_>>()
        .join(", ")
}
