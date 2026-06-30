use std::path::PathBuf;

use image::{Rgba, RgbaImage};
use imageproc::{drawing::draw_hollow_rect_mut, rect::Rect};
use ndarray::Array4;
use serde::Serialize;

pub mod model;
pub mod postprocess;
pub mod preprocess;

pub use model::{OrtDocLayout, RawTensorDump, detect_pdf_to_output_dir};
pub use postprocess::{cxcywh_to_xyxy, parse_detr_outputs, parse_paddle_fetch_output};
pub use preprocess::image_to_nchw_rgb;

// Label definitions live outside `layout` so preprocessing, postprocessing and callers can share
// the model vocabulary without depending on the detector module internals.
use crate::PPDocLayoutV3Label;

/// Upstream Hugging Face URL for the PP-DocLayoutV3 ONNX model file.
pub const MODEL_URL: &str =
    "https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx/resolve/main/inference.onnx";
/// Square model input size used by the upstream `inference.yml` preprocessing config.
pub const TARGET_SIZE: u32 = 800;
/// Default PDF render resolution used before layout detection.
pub const DEFAULT_DPI: f32 = 96.0;
/// Default confidence threshold for keeping layout detections.
pub const DEFAULT_THRESHOLD: f32 = 0.5;
/// Default directory for annotated page images and JSON detection output.
pub const DEFAULT_OUTPUT_DIR: &str = "output";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct OriginalSize {
    pub width: u32,
    pub height: u32,
}

impl OriginalSize {
    /// Builds an original image size value from pixel dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

#[derive(Debug, Clone)]
pub struct PreprocessedImage {
    pub tensor: Array4<f32>,
    pub im_shape: ndarray::Array2<f32>,
    pub scale_factor: ndarray::Array2<f32>,
    pub original_size: OriginalSize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Detection {
    pub class_id: usize,
    pub label: PPDocLayoutV3Label,
    pub score: f32,
    pub bbox: [f32; 4],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<usize>,
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

#[derive(Debug, Clone, Serialize)]
pub struct PdfPageOutputFile {
    pub page_number: u32,
    pub image_path: PathBuf,
    pub json_path: PathBuf,
    pub detections: usize,
}

/// Draws colored layout detection boxes onto a rendered page image.
pub fn annotate_page_rgba(image: &mut RgbaImage, detections: &[Detection]) {
    let width = image.width() as i32;
    let height = image.height() as i32;
    if width == 0 || height == 0 {
        return;
    }

    for detection in detections {
        let [x1, y1, x2, y2] = detection.bbox;
        let left = (x1.round() as i32).clamp(0, width - 1);
        let top = (y1.round() as i32).clamp(0, height - 1);
        let right = (x2.round() as i32).clamp(0, width - 1);
        let bottom = (y2.round() as i32).clamp(0, height - 1);
        if right < left || bottom < top {
            continue;
        }

        let rect =
            Rect::at(left, top).of_size((right - left + 1) as u32, (bottom - top + 1) as u32);
        draw_hollow_rect_mut(image, rect, Rgba(detection.label.debug_color_rgba()));
    }
}
