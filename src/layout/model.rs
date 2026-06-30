use std::{fs, path::Path};

use image::DynamicImage;
use ort::{session::Session, value::TensorRef};
use serde::Serialize;
use tracing::{debug, info};

use super::{
    DEFAULT_DPI, DEFAULT_OUTPUT_DIR, DEFAULT_THRESHOLD, Detection, PdfPageDetections,
    PdfPageOutputFile, PreprocessedImage, annotate_page_rgba, parse_detr_outputs,
    parse_paddle_fetch_output,
};
use crate::pdf::{PdfiumSession, encode_png_rgba, output_path_for_page};
use crate::{Error, PPDocLayoutV3Label, Result, ResultExt};

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

        info!(model = %model_path.display(), ?intra_threads, "loading ONNX model");
        let mut builder = Session::builder().context("create ONNX Runtime session builder")?;
        if let Some(threads) = intra_threads {
            builder = builder
                .with_intra_threads(threads)
                .context("configure intra-op thread count")?;
        }

        let session = builder
            .commit_from_file(model_path)
            .with_context(|| format!("load ONNX model {}", model_path.display()))?;
        info!(model = %model_path.display(), "loaded ONNX model");
        Ok(Self { session })
    }

    /// Runs layout detection on a single image file.
    pub fn detect_image_path(
        &mut self,
        image_path: impl AsRef<Path>,
        threshold: f32,
    ) -> Result<Vec<Detection>> {
        let image_path = image_path.as_ref();
        info!(path = %image_path.display(), threshold, "detecting layout in image");
        let image = image::open(image_path)
            .with_context(|| format!("open input image {}", image_path.display()))?;
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
            debug!(
                page_number = rendered.page_number,
                "running layout detection for PDF page"
            );
            let rgb = DynamicImage::ImageRgba8(rendered.rgba.clone()).to_rgb8();
            let input = PreprocessedImage::try_from(&rgb)?;
            let detections = self.detect_preprocessed(input, DEFAULT_THRESHOLD)?;
            debug!(
                page_number = rendered.page_number,
                detections = detections.len(),
                "detected PDF page layouts"
            );
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
        info!(path = %image_path.display(), max_values, "dumping raw model outputs");
        let image = image::open(image_path)
            .with_context(|| format!("open input image {}", image_path.display()))?;
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
            .ok_or_else(|| Error::ModelOutput {
                message: format!(
                    "could not find DETR logits output; available outputs: {}",
                    format_output_shapes(&arrays)
                ),
            })?;
        let (_, _, boxes) = arrays
            .iter()
            .find(|(_, shape, _)| shape.len() == 3 && shape[0] == 1 && shape[2] == 4)
            .ok_or_else(|| Error::ModelOutput {
                message: format!(
                    "could not find DETR boxes output; available outputs: {}",
                    format_output_shapes(&arrays)
                ),
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
            .context("create ONNX tensor from preprocessed image")?;
        let im_shape_tensor =
            TensorRef::from_array_view(&input.im_shape).context("create ONNX im_shape tensor")?;
        let scale_factor_tensor = TensorRef::from_array_view(&input.scale_factor)
            .context("create ONNX scale_factor tensor")?;
        debug!("running ONNX inference");
        self.session
            .run(ort::inputs! {
                "im_shape" => im_shape_tensor,
                "image" => image_tensor,
                "scale_factor" => scale_factor_tensor,
            })
            .context("run ONNX inference")
    }

    /// Detects layouts in a PDF and writes annotated PNG plus JSON per page.
    pub fn detect_pdf_to_output_dir(
        &mut self,
        pdf_path: impl AsRef<Path>,
        output_dir: impl AsRef<Path>,
    ) -> Result<Vec<PdfPageOutputFile>> {
        let output_dir = output_dir.as_ref();
        fs::create_dir_all(output_dir)
            .with_context(|| format!("create output dir {}", output_dir.display()))?;

        info!(output_dir = %output_dir.display(), "detecting PDF layouts to output directory");
        let pdfium = PdfiumSession::new();
        let pdf = pdfium.open_document(pdf_path)?;
        let mut outputs = Vec::new();
        pdf.visit_rendered_pages(DEFAULT_DPI, |mut rendered| {
            debug!(
                page_number = rendered.page_number,
                "processing rendered PDF page"
            );
            let rgb = DynamicImage::ImageRgba8(rendered.rgba.clone()).to_rgb8();
            let input = PreprocessedImage::try_from(&rgb)?;
            let detections = self.detect_preprocessed(input, DEFAULT_THRESHOLD)?;
            annotate_page_rgba(&mut rendered.rgba, &detections);

            let image_path = output_path_for_page(output_dir, rendered.page_number, "png");
            let json_path = output_path_for_page(output_dir, rendered.page_number, "json");
            let png_bytes = encode_png_rgba(&rendered.rgba)?;
            fs::write(&image_path, png_bytes)
                .with_context(|| format!("write {}", image_path.display()))?;

            let page_output = PdfPageDetections {
                page_number: rendered.page_number,
                width: rendered.width,
                height: rendered.height,
                page_width: rendered.page_width,
                page_height: rendered.page_height,
                dpi: DEFAULT_DPI,
                detections,
            };
            let json = serde_json::to_vec_pretty(&page_output)
                .with_context(|| format!("serialize page {} JSON", rendered.page_number))?;
            fs::write(&json_path, json)
                .with_context(|| format!("write {}", json_path.display()))?;
            debug!(
                page_number = rendered.page_number,
                detections = page_output.detections.len(),
                image_path = %image_path.display(),
                json_path = %json_path.display(),
                "wrote PDF page outputs"
            );

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

    Err(Error::MissingModel {
        path: model_path.to_path_buf(),
    })
}

/// Formats model output tensor names and shapes for parser error messages.
fn format_output_shapes(outputs: &[(String, Vec<usize>, ndarray::ArrayD<f32>)]) -> String {
    outputs
        .iter()
        .map(|(name, shape, _)| format!("{name}: {shape:?}"))
        .collect::<Vec<_>>()
        .join(", ")
}
