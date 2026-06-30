use std::{
    fmt, fs,
    path::{Path, PathBuf},
    str::FromStr,
};

use anyhow::{Context, Result, anyhow, bail};
use image::{
    ColorType, DynamicImage, ImageEncoder, RgbImage, RgbaImage, codecs::png::PngEncoder,
    imageops::FilterType,
};
use ndarray::{Array2, Array3, Array4, ArrayD, Ix2, Ix3};
use ort::{ep, session::Session, value::TensorRef};
use pdfium::Library;
use serde::Serialize;

pub const MODEL_URL: &str =
    "https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx/resolve/main/inference.onnx";
pub const TARGET_SIZE: u32 = 800;
pub const DEFAULT_DPI: f32 = 96.0;
pub const DEFAULT_THRESHOLD: f32 = 0.5;
pub const DEFAULT_OUTPUT_DIR: &str = "output";

pub const PP_DOCLAYOUT_V3_LABELS: [PPDocLayoutV3Label; 25] = [
    PPDocLayoutV3Label::Abstract,
    PPDocLayoutV3Label::Algorithm,
    PPDocLayoutV3Label::AsideText,
    PPDocLayoutV3Label::Chart,
    PPDocLayoutV3Label::Content,
    PPDocLayoutV3Label::DisplayFormula,
    PPDocLayoutV3Label::DocTitle,
    PPDocLayoutV3Label::FigureTitle,
    PPDocLayoutV3Label::Footer,
    PPDocLayoutV3Label::FooterImage,
    PPDocLayoutV3Label::Footnote,
    PPDocLayoutV3Label::FormulaNumber,
    PPDocLayoutV3Label::Header,
    PPDocLayoutV3Label::HeaderImage,
    PPDocLayoutV3Label::Image,
    PPDocLayoutV3Label::InlineFormula,
    PPDocLayoutV3Label::Number,
    PPDocLayoutV3Label::ParagraphTitle,
    PPDocLayoutV3Label::Reference,
    PPDocLayoutV3Label::ReferenceContent,
    PPDocLayoutV3Label::Seal,
    PPDocLayoutV3Label::Table,
    PPDocLayoutV3Label::Text,
    PPDocLayoutV3Label::VerticalText,
    PPDocLayoutV3Label::VisionFootnote,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PPDocLayoutV3Label {
    Abstract,
    Algorithm,
    AsideText,
    Chart,
    Content,
    DisplayFormula,
    DocTitle,
    FigureTitle,
    Footer,
    FooterImage,
    Footnote,
    FormulaNumber,
    Header,
    HeaderImage,
    Image,
    InlineFormula,
    Number,
    ParagraphTitle,
    Reference,
    ReferenceContent,
    Seal,
    Table,
    Text,
    VerticalText,
    VisionFootnote,
}

impl PPDocLayoutV3Label {
    pub const fn class_count() -> usize {
        PP_DOCLAYOUT_V3_LABELS.len()
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Abstract => "abstract",
            Self::Algorithm => "algorithm",
            Self::AsideText => "aside_text",
            Self::Chart => "chart",
            Self::Content => "content",
            Self::DisplayFormula => "display_formula",
            Self::DocTitle => "doc_title",
            Self::FigureTitle => "figure_title",
            Self::Footer => "footer",
            Self::FooterImage => "footer_image",
            Self::Footnote => "footnote",
            Self::FormulaNumber => "formula_number",
            Self::Header => "header",
            Self::HeaderImage => "header_image",
            Self::Image => "image",
            Self::InlineFormula => "inline_formula",
            Self::Number => "number",
            Self::ParagraphTitle => "paragraph_title",
            Self::Reference => "reference",
            Self::ReferenceContent => "reference_content",
            Self::Seal => "seal",
            Self::Table => "table",
            Self::Text => "text",
            Self::VerticalText => "vertical_text",
            Self::VisionFootnote => "vision_footnote",
        }
    }

    pub fn debug_color_rgba(self) -> [u8; 4] {
        match self {
            Self::Abstract => [0x7C, 0x4D, 0xFF, 255],
            Self::Algorithm => [0x5C, 0x6B, 0xC0, 255],
            Self::AsideText => [0x26, 0xA6, 0x9A, 255],
            Self::Chart => [0xEF, 0x6C, 0x00, 255],
            Self::Content => [0x43, 0xA0, 0x47, 255],
            Self::DisplayFormula => [0xAB, 0x47, 0xBC, 255],
            Self::DocTitle => [0xD8, 0x1B, 0x60, 255],
            Self::FigureTitle => [0x00, 0x89, 0x7B, 255],
            Self::Footer => [0x8D, 0x6E, 0x63, 255],
            Self::FooterImage => [0xA1, 0x88, 0x7F, 255],
            Self::Footnote => [0xF4, 0x43, 0x36, 255],
            Self::FormulaNumber => [0x8E, 0x24, 0xAA, 255],
            Self::Header => [0xFF, 0x8F, 0x00, 255],
            Self::HeaderImage => [0xF9, 0xA8, 0x25, 255],
            Self::Image => [0x9E, 0x9E, 0x9E, 255],
            Self::InlineFormula => [0x7B, 0x1F, 0xA2, 255],
            Self::Number => [0x39, 0x49, 0xAB, 255],
            Self::ParagraphTitle => [0x1E, 0x88, 0xE5, 255],
            Self::Reference => [0x00, 0xAC, 0xC1, 255],
            Self::ReferenceContent => [0x00, 0x96, 0x88, 255],
            Self::Seal => [0xC6, 0x28, 0x28, 255],
            Self::Table => [0x00, 0x96, 0x88, 255],
            Self::Text => [0x43, 0xA0, 0x47, 255],
            Self::VerticalText => [0x6D, 0x4C, 0x41, 255],
            Self::VisionFootnote => [0xEC, 0x40, 0x7A, 255],
        }
    }
}

impl TryFrom<usize> for PPDocLayoutV3Label {
    type Error = PPDocLayoutV3LabelError;

    fn try_from(value: usize) -> std::result::Result<Self, Self::Error> {
        PP_DOCLAYOUT_V3_LABELS
            .get(value)
            .copied()
            .ok_or(PPDocLayoutV3LabelError::UnknownClassId(value))
    }
}

impl FromStr for PPDocLayoutV3Label {
    type Err = PPDocLayoutV3LabelError;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        PP_DOCLAYOUT_V3_LABELS
            .iter()
            .copied()
            .find(|label| label.as_str() == value)
            .ok_or_else(|| PPDocLayoutV3LabelError::UnknownLabel(value.to_string()))
    }
}

impl fmt::Display for PPDocLayoutV3Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PPDocLayoutV3LabelError {
    UnknownClassId(usize),
    UnknownLabel(String),
}

impl fmt::Display for PPDocLayoutV3LabelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownClassId(class_id) => {
                write!(f, "unknown PP-DocLayoutV3 class id: {class_id}")
            }
            Self::UnknownLabel(label) => write!(f, "unknown PP-DocLayoutV3 label: {label}"),
        }
    }
}

impl std::error::Error for PPDocLayoutV3LabelError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct OriginalSize {
    pub width: u32,
    pub height: u32,
}

impl OriginalSize {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

#[derive(Debug, Clone)]
pub struct PreprocessedImage {
    pub tensor: Array4<f32>,
    pub im_shape: Array2<f32>,
    pub scale_factor: Array2<f32>,
    pub original_size: OriginalSize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Detection {
    pub class_id: usize,
    pub label: PPDocLayoutV3Label,
    pub score: f32,
    pub bbox: [f32; 4],
}

#[derive(Debug, Clone, Serialize)]
pub struct PdfPageDetections {
    pub page_number: u32,
    pub width: u32,
    pub height: u32,
    pub page_width: f32,
    pub page_height: f32,
    pub dpi: f32,
    pub detections: Vec<Detection>,
}

#[derive(Debug, Clone)]
pub struct RenderedPdfPage {
    pub page_number: u32,
    pub width: u32,
    pub height: u32,
    pub page_width: f32,
    pub page_height: f32,
    pub rgb: RgbImage,
    pub rgba: RgbaImage,
}

#[derive(Debug, Clone, Serialize)]
pub struct PdfPageOutputFile {
    pub page_number: u32,
    pub image_path: PathBuf,
    pub json_path: PathBuf,
    pub detections: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub inputs: Vec<TensorInfo>,
    pub outputs: Vec<TensorInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RawTensorDump {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

pub struct OrtDocLayout {
    session: Session,
}

impl OrtDocLayout {
    pub fn new(model_path: impl AsRef<Path>, intra_threads: Option<usize>) -> Result<Self> {
        let model_path = model_path.as_ref();
        ensure_model_exists(model_path)?;

        let mut builder = Session::builder()
            .map_err(|error| anyhow!("create ONNX Runtime session builder: {error}"))?;
        builder = builder
            .with_execution_providers([ep::WebGPU::default().build()])
            .map_err(|error| {
                anyhow!("configure ONNX Runtime WebGPU execution provider: {error}")
            })?;
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

    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            inputs: self
                .session
                .inputs()
                .iter()
                .map(|input| TensorInfo {
                    name: input.name().to_string(),
                    dtype: format!("{:?}", input.dtype()),
                })
                .collect(),
            outputs: self
                .session
                .outputs()
                .iter()
                .map(|output| TensorInfo {
                    name: output.name().to_string(),
                    dtype: format!("{:?}", output.dtype()),
                })
                .collect(),
        }
    }

    pub fn detect_image_path(
        &mut self,
        image_path: impl AsRef<Path>,
        threshold: f32,
    ) -> Result<Vec<Detection>> {
        let image_path = image_path.as_ref();
        let image = image::open(image_path)
            .with_context(|| format!("open input image {}", image_path.display()))?;
        let input = preprocess_image(&image);
        self.detect_preprocessed(input, threshold)
    }

    pub fn detect_pdf_path(
        &mut self,
        pdf_path: impl AsRef<Path>,
    ) -> Result<Vec<PdfPageDetections>> {
        let rendered_pages = render_pdf_pages(pdf_path, DEFAULT_DPI)?;
        let mut pages = Vec::with_capacity(rendered_pages.len());

        for rendered in rendered_pages {
            let input = preprocess_rgb_image(&rendered.rgb);
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
        }

        Ok(pages)
    }

    pub fn dump_image_outputs(
        &mut self,
        image_path: impl AsRef<Path>,
        max_values: usize,
    ) -> Result<Vec<RawTensorDump>> {
        let image_path = image_path.as_ref();
        let image = image::open(image_path)
            .with_context(|| format!("open input image {}", image_path.display()))?;
        let input = preprocess_image(&image);
        let image_tensor = TensorRef::from_array_view(&input.tensor)
            .map_err(|error| anyhow!("create ONNX tensor from preprocessed image: {error}"))?;
        let im_shape_tensor = TensorRef::from_array_view(&input.im_shape)
            .map_err(|error| anyhow!("create ONNX im_shape tensor: {error}"))?;
        let scale_factor_tensor = TensorRef::from_array_view(&input.scale_factor)
            .map_err(|error| anyhow!("create ONNX scale_factor tensor: {error}"))?;
        let outputs = self
            .session
            .run(ort::inputs! {
                "im_shape" => im_shape_tensor,
                "image" => image_tensor,
                "scale_factor" => scale_factor_tensor,
            })
            .map_err(|error| anyhow!("run ONNX inference: {error}"))?;

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

    pub fn detect_preprocessed(
        &mut self,
        input: PreprocessedImage,
        threshold: f32,
    ) -> Result<Vec<Detection>> {
        let image_tensor = TensorRef::from_array_view(&input.tensor)
            .map_err(|error| anyhow!("create ONNX tensor from preprocessed image: {error}"))?;
        let im_shape_tensor = TensorRef::from_array_view(&input.im_shape)
            .map_err(|error| anyhow!("create ONNX im_shape tensor: {error}"))?;
        let scale_factor_tensor = TensorRef::from_array_view(&input.scale_factor)
            .map_err(|error| anyhow!("create ONNX scale_factor tensor: {error}"))?;
        let outputs = self
            .session
            .run(ort::inputs! {
                "im_shape" => im_shape_tensor,
                "image" => image_tensor,
                "scale_factor" => scale_factor_tensor,
            })
            .map_err(|error| anyhow!("run ONNX inference: {error}"))?;

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
            return parse_paddle_fetch_output(output, input.original_size, threshold);
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
            input.original_size,
            threshold,
        )
    }
}

pub fn inspect_model(model_path: impl AsRef<Path>) -> Result<ModelInfo> {
    let runner = OrtDocLayout::new(model_path, None)?;
    Ok(runner.model_info())
}

pub fn detect_pdf_to_output_dir(
    pdf_path: impl AsRef<Path>,
    model_path: impl AsRef<Path>,
) -> Result<Vec<PdfPageOutputFile>> {
    let mut detector = OrtDocLayout::new(model_path, None)?;
    detector.detect_pdf_to_output_dir(pdf_path, DEFAULT_OUTPUT_DIR)
}

impl OrtDocLayout {
    pub fn detect_pdf_to_output_dir(
        &mut self,
        pdf_path: impl AsRef<Path>,
        output_dir: impl AsRef<Path>,
    ) -> Result<Vec<PdfPageOutputFile>> {
        let output_dir = output_dir.as_ref();
        fs::create_dir_all(output_dir)
            .with_context(|| format!("create output dir {}", output_dir.display()))?;

        let rendered_pages = render_pdf_pages(pdf_path, DEFAULT_DPI)?;
        let mut outputs = Vec::with_capacity(rendered_pages.len());
        for mut rendered in rendered_pages {
            let input = preprocess_rgb_image(&rendered.rgb);
            let detections = self.detect_preprocessed(input, DEFAULT_THRESHOLD)?;
            annotate_page_rgba(&mut rendered.rgba, &detections);

            let image_path = output_path_for_page(output_dir, rendered.page_number, "png");
            let json_path = output_path_for_page(output_dir, rendered.page_number, "json");
            let png_bytes = encode_png_rgba(&rendered.rgba)
                .with_context(|| format!("encode page {} PNG", rendered.page_number))?;
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

            outputs.push(PdfPageOutputFile {
                page_number: rendered.page_number,
                image_path,
                json_path,
                detections: page_output.detections.len(),
            });
        }

        Ok(outputs)
    }
}

pub fn output_path_for_page(
    output_dir: impl AsRef<Path>,
    page_number: u32,
    extension: &str,
) -> PathBuf {
    output_dir
        .as_ref()
        .join(format!("page-{page_number:04}.{extension}"))
}

pub fn render_pdf_pages(pdf_path: impl AsRef<Path>, dpi: f32) -> Result<Vec<RenderedPdfPage>> {
    let pdf_path = pdf_path.as_ref();
    if dpi <= 0.0 {
        bail!("dpi must be greater than zero, got {dpi}");
    }

    let pdf_bytes =
        fs::read(pdf_path).with_context(|| format!("read PDF {}", pdf_path.display()))?;
    let lib = Library::init();
    let document = lib
        .load_document_from_bytes(&pdf_bytes, None)
        .map_err(|error| anyhow!("load PDF {}: {error}", pdf_path.display()))?;
    let page_count = usize::try_from(document.page_count())
        .map_err(|error| anyhow!("convert PDF page count: {error}"))?;
    let mut pages = Vec::with_capacity(page_count);

    for page_index in 0..page_count {
        let page_number = page_index as u32 + 1;
        let page = document
            .page(i32::try_from(page_index).expect("page index should fit into i32"))
            .map_err(|error| anyhow!("load PDF page {page_number}: {error}"))?;
        let page_width = page.width();
        let page_height = page.height();
        let bitmap = page
            .render(dpi)
            .map_err(|error| anyhow!("render PDF page {page_number}: {error}"))?;
        let width = bitmap.width() as u32;
        let height = bitmap.height() as u32;
        let rgb = bitmap.to_rgb();
        let rgb = RgbImage::from_raw(width, height, rgb).ok_or_else(|| {
            anyhow!("PDF page {page_number} rendered RGB buffer does not match {width}x{height}")
        })?;
        let rgba = bitmap.to_rgba();
        let rgba = RgbaImage::from_raw(width, height, rgba).ok_or_else(|| {
            anyhow!("PDF page {page_number} rendered RGBA buffer does not match {width}x{height}")
        })?;

        pages.push(RenderedPdfPage {
            page_number,
            width,
            height,
            page_width,
            page_height,
            rgb,
            rgba,
        });
    }

    Ok(pages)
}

pub fn preprocess_image(image: &DynamicImage) -> PreprocessedImage {
    preprocess_rgb_image(&image.to_rgb8())
}

pub fn preprocess_rgb_image(image: &RgbImage) -> PreprocessedImage {
    let original_size = OriginalSize::new(image.width(), image.height());
    let resized = image::imageops::resize(image, TARGET_SIZE, TARGET_SIZE, FilterType::Triangle);
    let scale_y = TARGET_SIZE as f32 / image.height() as f32;
    let scale_x = TARGET_SIZE as f32 / image.width() as f32;
    PreprocessedImage {
        tensor: image_to_nchw_rgb(&resized),
        im_shape: Array2::from_shape_vec((1, 2), vec![TARGET_SIZE as f32, TARGET_SIZE as f32])
            .expect("static im_shape should match shape"),
        scale_factor: Array2::from_shape_vec((1, 2), vec![scale_y, scale_x])
            .expect("static scale_factor should match shape"),
        original_size,
    }
}

pub fn image_to_nchw_rgb(image: &RgbImage) -> Array4<f32> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let mut tensor = Array4::<f32>::zeros((1, 3, height, width));

    for (x, y, pixel) in image.enumerate_pixels() {
        let x = x as usize;
        let y = y as usize;
        tensor[[0, 0, y, x]] = f32::from(pixel[0]) / 255.0;
        tensor[[0, 1, y, x]] = f32::from(pixel[1]) / 255.0;
        tensor[[0, 2, y, x]] = f32::from(pixel[2]) / 255.0;
    }

    tensor
}

pub fn encode_png_rgba(image: &RgbaImage) -> Result<Vec<u8>> {
    let mut png_bytes = Vec::new();
    let encoder = PngEncoder::new(&mut png_bytes);
    encoder
        .write_image(
            image.as_raw(),
            image.width(),
            image.height(),
            ColorType::Rgba8.into(),
        )
        .context("write RGBA PNG data")?;
    Ok(png_bytes)
}

pub fn annotate_page_rgba(image: &mut RgbaImage, detections: &[Detection]) {
    let width = image.width() as i32;
    let height = image.height() as i32;
    if width == 0 || height == 0 {
        return;
    }

    let rgba = image.as_mut();
    for detection in detections {
        let [x1, y1, x2, y2] = detection.bbox;
        let rect = Rect {
            left: x1.round() as i32,
            top: y1.round() as i32,
            right: x2.round() as i32,
            bottom: y2.round() as i32,
        };
        draw_rect_outline(
            rgba,
            width,
            height,
            rect,
            detection.label.debug_color_rgba(),
        );
    }
}

#[derive(Debug, Clone, Copy)]
struct Rect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

fn draw_rect_outline(
    rgba: &mut [u8],
    image_width: i32,
    image_height: i32,
    rect: Rect,
    color: [u8; 4],
) {
    if rect.right < rect.left || rect.bottom < rect.top {
        return;
    }

    let left = rect.left.clamp(0, image_width - 1);
    let top = rect.top.clamp(0, image_height - 1);
    let right = rect.right.clamp(0, image_width - 1);
    let bottom = rect.bottom.clamp(0, image_height - 1);

    for thickness in 0..2 {
        draw_horizontal_line(
            rgba,
            image_width,
            image_height,
            left,
            right,
            top + thickness,
            color,
        );
        draw_horizontal_line(
            rgba,
            image_width,
            image_height,
            left,
            right,
            bottom - thickness,
            color,
        );
        draw_vertical_line(
            rgba,
            image_width,
            image_height,
            left + thickness,
            top,
            bottom,
            color,
        );
        draw_vertical_line(
            rgba,
            image_width,
            image_height,
            right - thickness,
            top,
            bottom,
            color,
        );
    }
}

fn draw_horizontal_line(
    rgba: &mut [u8],
    image_width: i32,
    image_height: i32,
    left: i32,
    right: i32,
    y: i32,
    color: [u8; 4],
) {
    if y < 0 || y >= image_height {
        return;
    }

    for x in left.max(0)..=right.min(image_width - 1) {
        put_pixel(rgba, image_width, x, y, color);
    }
}

fn draw_vertical_line(
    rgba: &mut [u8],
    image_width: i32,
    image_height: i32,
    x: i32,
    top: i32,
    bottom: i32,
    color: [u8; 4],
) {
    if x < 0 || x >= image_width {
        return;
    }

    for y in top.max(0)..=bottom.min(image_height - 1) {
        put_pixel(rgba, image_width, x, y, color);
    }
}

fn put_pixel(rgba: &mut [u8], image_width: i32, x: i32, y: i32, color: [u8; 4]) {
    let offset = ((y * image_width + x) * 4) as usize;
    rgba[offset..offset + 4].copy_from_slice(&color);
}

pub fn parse_detr_outputs(
    logits: ArrayD<f32>,
    boxes: ArrayD<f32>,
    original_size: OriginalSize,
    threshold: f32,
) -> Result<Vec<Detection>> {
    if !(0.0..=1.0).contains(&threshold) {
        bail!("threshold must be in [0, 1], got {threshold}");
    }

    let logits = logits
        .into_dimensionality::<Ix3>()
        .context("logits output should have shape [1, queries, classes]")?;
    let boxes = boxes
        .into_dimensionality::<Ix3>()
        .context("boxes output should have shape [1, queries, 4]")?;

    let [logit_batch, query_count, class_count]: [usize; 3] = logits
        .shape()
        .try_into()
        .expect("Ix3 shape should have three dimensions");
    let [box_batch, box_count, box_dims]: [usize; 3] = boxes
        .shape()
        .try_into()
        .expect("Ix3 shape should have three dimensions");

    if logit_batch != 1 || box_batch != 1 {
        bail!(
            "only batch size 1 is supported, got logits batch {logit_batch}, boxes batch {box_batch}"
        );
    }
    if query_count != box_count || box_dims != 4 {
        bail!(
            "logits and boxes shape mismatch: logits {:?}, boxes {:?}",
            logits.shape(),
            boxes.shape()
        );
    }

    let foreground_classes = class_count.min(PPDocLayoutV3Label::class_count());
    let mut detections = Vec::new();
    for query in 0..query_count {
        let mut best_class = 0;
        let mut best_logit = f32::NEG_INFINITY;
        for class_id in 0..foreground_classes {
            let logit = logits[[0, query, class_id]];
            if logit > best_logit {
                best_logit = logit;
                best_class = class_id;
            }
        }

        let score = softmax_class_score(&logits, query, best_class);
        if score < threshold {
            continue;
        }

        let bbox = cxcywh_to_xyxy(
            [
                boxes[[0, query, 0]],
                boxes[[0, query, 1]],
                boxes[[0, query, 2]],
                boxes[[0, query, 3]],
            ],
            original_size,
        );
        detections.push(Detection {
            class_id: best_class,
            label: PPDocLayoutV3Label::try_from(best_class)
                .expect("best class is bounded by foreground_classes"),
            score,
            bbox,
        });
    }

    detections.sort_by(|left, right| right.score.total_cmp(&left.score));
    Ok(detections)
}

pub fn parse_paddle_fetch_output(
    output: ArrayD<f32>,
    original_size: OriginalSize,
    threshold: f32,
) -> Result<Vec<Detection>> {
    if !(0.0..=1.0).contains(&threshold) {
        bail!("threshold must be in [0, 1], got {threshold}");
    }

    let output = output
        .into_dimensionality::<Ix2>()
        .context("Paddle fetch output should have shape [boxes, 6 or 7]")?;
    let columns = output.shape()[1];
    if columns != 6 && columns != 7 {
        bail!("Paddle fetch output should have 6 or 7 columns, got {columns}");
    }

    let mut detections = Vec::new();
    for row in output.rows() {
        let class_id = row[0].round() as usize;
        let score = row[1];
        let Ok(label) = PPDocLayoutV3Label::try_from(class_id) else {
            continue;
        };
        if score < threshold {
            continue;
        }

        detections.push(Detection {
            class_id,
            label,
            score,
            bbox: clamp_xyxy([row[2], row[3], row[4], row[5]], original_size),
        });
    }

    detections.sort_by(|left, right| right.score.total_cmp(&left.score));
    Ok(detections)
}

pub fn cxcywh_to_xyxy(box_values: [f32; 4], original_size: OriginalSize) -> [f32; 4] {
    let [cx, cy, width, height] = box_values;
    let normalized = box_values
        .iter()
        .all(|value| value.is_finite() && *value >= -0.01 && *value <= 1.5);
    let scale_x = if normalized {
        original_size.width as f32
    } else {
        original_size.width as f32 / TARGET_SIZE as f32
    };
    let scale_y = if normalized {
        original_size.height as f32
    } else {
        original_size.height as f32 / TARGET_SIZE as f32
    };

    let x1 = (cx - width / 2.0) * scale_x;
    let y1 = (cy - height / 2.0) * scale_y;
    let x2 = (cx + width / 2.0) * scale_x;
    let y2 = (cy + height / 2.0) * scale_y;

    [
        x1.clamp(0.0, original_size.width as f32),
        y1.clamp(0.0, original_size.height as f32),
        x2.clamp(0.0, original_size.width as f32),
        y2.clamp(0.0, original_size.height as f32),
    ]
}

fn softmax_class_score(logits: &Array3<f32>, query: usize, class_id: usize) -> f32 {
    let class_count = logits.shape()[2];
    let mut max_logit = f32::NEG_INFINITY;
    for class in 0..class_count {
        max_logit = max_logit.max(logits[[0, query, class]]);
    }

    let mut denominator = 0.0;
    for class in 0..class_count {
        denominator += (logits[[0, query, class]] - max_logit).exp();
    }

    ((logits[[0, query, class_id]] - max_logit).exp()) / denominator
}

fn clamp_xyxy(bbox: [f32; 4], original_size: OriginalSize) -> [f32; 4] {
    [
        bbox[0].clamp(0.0, original_size.width as f32),
        bbox[1].clamp(0.0, original_size.height as f32),
        bbox[2].clamp(0.0, original_size.width as f32),
        bbox[3].clamp(0.0, original_size.height as f32),
    ]
}

fn ensure_model_exists(model_path: &Path) -> Result<()> {
    if model_path.exists() {
        return Ok(());
    }

    bail!(
        "model file does not exist: {}\ndownload it with:\n  curl -L {} -o {}",
        model_path.display(),
        MODEL_URL,
        model_path.display()
    );
}

fn format_output_shapes(outputs: &[(String, Vec<usize>, ndarray::ArrayD<f32>)]) -> String {
    outputs
        .iter()
        .map(|(name, shape, _)| format!("{name}: {shape:?}"))
        .collect::<Vec<_>>()
        .join(", ")
}
