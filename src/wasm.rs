use image::{DynamicImage, RgbaImage};
use js_sys::{Array, Float32Array, JsString, Object, Reflect, Uint8ClampedArray};
use serde::Serialize;
use tracing::{debug, info};
use wasm_bindgen::{JsCast, prelude::*};

use crate::{
    DEFAULT_THRESHOLD, Detection, Error, OriginalSize, PPDocLayoutV3Label, PreprocessedImage,
    Result, ResultExt, annotate_page_rgba, label_info, parse_detr_outputs,
    parse_paddle_fetch_output,
};

/// Browser-facing page result returned after running layout detection.
#[derive(Debug, Clone, Serialize)]
pub struct WasmPageDetections {
    pub width: u32,
    pub height: u32,
    pub detections: Vec<Detection>,
}

/// Browser ONNX Runtime-backed detector using the `ort-web` alternative backend.
#[wasm_bindgen]
pub struct BrowserDocLayout {
    session: InferenceSession,
    threshold: f32,
}

#[wasm_bindgen]
extern "C" {
    /// Browser ORT Web session object loaded by `ort-web`.
    #[wasm_bindgen(js_namespace = ort, js_name = InferenceSession)]
    type InferenceSession;

    /// Creates an ORT Web inference session from a browser-visible model URL.
    #[wasm_bindgen(catch, js_namespace = ort, static_method_of = InferenceSession, js_name = create)]
    async fn create(
        model_url: &str,
        options: JsValue,
    ) -> std::result::Result<InferenceSession, JsValue>;

    /// Runs ORT Web inference with JS-owned tensors.
    #[wasm_bindgen(catch, structural, method, js_name = run)]
    async fn run(this: &InferenceSession, feeds: JsValue) -> std::result::Result<JsValue, JsValue>;

    /// Browser ORT Web tensor object used for JS-owned input and output buffers.
    #[wasm_bindgen(js_namespace = ort, js_name = Tensor)]
    type JsTensor;

    /// Creates a JS-owned ORT Web tensor.
    #[wasm_bindgen(constructor, catch, js_namespace = ort, js_class = Tensor)]
    fn new(dtype: &str, data: JsValue, dims: JsValue) -> std::result::Result<JsTensor, JsValue>;

    /// Returns CPU-accessible tensor data when it is already available.
    #[wasm_bindgen(structural, catch, method, getter, js_name = data)]
    fn data(this: &JsTensor) -> std::result::Result<JsValue, JsValue>;

    /// Downloads tensor data from the active ORT Web execution provider.
    #[wasm_bindgen(structural, catch, method, js_name = getData)]
    async fn get_data(this: &JsTensor) -> std::result::Result<JsValue, JsValue>;

    /// Returns the tensor dimensions reported by ORT Web.
    #[wasm_bindgen(structural, method, getter, js_name = dims)]
    fn dims(this: &JsTensor) -> Vec<i32>;
}

/// Installs browser diagnostics when the WASM module is initialized.
#[wasm_bindgen(start)]
pub fn start() {
    init_browser_diagnostics();
}

#[wasm_bindgen]
impl BrowserDocLayout {
    /// Loads ONNX Runtime Web and creates a PP-DocLayout session from a model URL.
    #[wasm_bindgen(js_name = load)]
    pub async fn load(
        model_url: String,
        ort_dist_base_url: Option<String>,
    ) -> std::result::Result<BrowserDocLayout, JsValue> {
        init_browser_diagnostics();
        init_ort_web(ort_dist_base_url).await?;

        info!(%model_url, "loading browser ONNX model");
        let session = InferenceSession::create(&model_url, browser_session_options())
            .await
            .map_err(|error| error)?;
        info!("loaded browser ONNX model");

        Ok(Self {
            session,
            threshold: DEFAULT_THRESHOLD,
        })
    }

    /// Returns the active confidence threshold.
    #[wasm_bindgen(getter)]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Sets the confidence threshold used by `detectRgba`.
    #[wasm_bindgen(setter)]
    pub fn set_threshold(&mut self, threshold: f32) -> std::result::Result<(), JsValue> {
        validate_threshold(threshold).map_err(js_error)?;
        self.threshold = threshold;
        Ok(())
    }

    /// Runs layout detection over an RGBA page image and returns page JSON.
    #[wasm_bindgen(js_name = detectRgba)]
    pub async fn detect_rgba(
        &mut self,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> std::result::Result<JsValue, JsValue> {
        let image = rgba_image_from_bytes(rgba, width, height).map_err(js_error)?;
        let rgb = DynamicImage::ImageRgba8(image).to_rgb8();
        let input = PreprocessedImage::try_from(&rgb).map_err(js_error)?;

        let detections = self
            .detect_preprocessed(input, self.threshold)
            .await
            .map_err(js_error)?;
        let page = WasmPageDetections {
            width,
            height,
            detections,
        };
        json_object(&page)
    }
}

/// Draws layout detections onto an RGBA page image and returns annotated RGBA bytes.
#[wasm_bindgen(js_name = annotateRgba)]
pub fn annotate_rgba(
    rgba: &[u8],
    width: u32,
    height: u32,
    detections: JsValue,
) -> std::result::Result<JsValue, JsValue> {
    let mut image = rgba_image_from_bytes(rgba, width, height).map_err(js_error)?;
    let detections: Vec<Detection> =
        serde_wasm_bindgen::from_value(detections).map_err(js_error)?;
    annotate_page_rgba(&mut image, &detections);
    rgba_array(image.as_raw())
}

/// Returns PP-DocLayoutV3 labels and annotation colors for browser legend rendering.
#[wasm_bindgen(js_name = labelLegend)]
pub fn label_legend() -> std::result::Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&label_info()).map_err(js_error)
}

impl BrowserDocLayout {
    /// Runs preprocessing output through ORT Web and parses model outputs into detections.
    async fn detect_preprocessed(
        &mut self,
        input: PreprocessedImage,
        threshold: f32,
    ) -> Result<Vec<Detection>> {
        let original_size = input.original_size;
        let mut arrays = Vec::new();
        let mut paddle_fetch_output = None;
        for (name, shape, array) in self.run_preprocessed_arrays(input).await? {
            if shape.len() == 2 && matches!(shape.get(1).copied(), Some(6 | 7)) {
                paddle_fetch_output = Some(array.clone().into_dyn());
            }
            arrays.push((name, shape, array));
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

    /// Converts preprocessed tensors into JS-owned ORT Web inputs and returns raw f32 outputs.
    async fn run_preprocessed_arrays(
        &mut self,
        input: PreprocessedImage,
    ) -> Result<Vec<(String, Vec<usize>, ndarray::ArrayD<f32>)>> {
        let image_tensor = js_tensor_from_f32(
            input.tensor.shape(),
            input.tensor.as_slice().ok_or_else(|| Error::InvalidInput {
                message: "preprocessed image tensor should be contiguous".to_string(),
            })?,
        )?;
        let im_shape_tensor = js_tensor_from_f32(
            input.im_shape.shape(),
            input
                .im_shape
                .as_slice()
                .ok_or_else(|| Error::InvalidInput {
                    message: "im_shape tensor should be contiguous".to_string(),
                })?,
        )?;
        let scale_factor_tensor = js_tensor_from_f32(
            input.scale_factor.shape(),
            input
                .scale_factor
                .as_slice()
                .ok_or_else(|| Error::InvalidInput {
                    message: "scale_factor tensor should be contiguous".to_string(),
                })?,
        )?;
        let feeds = Object::new();
        Reflect::set(&feeds, &JsValue::from_str("image"), &image_tensor).map_err(error_from_js)?;
        Reflect::set(&feeds, &JsValue::from_str("im_shape"), &im_shape_tensor)
            .map_err(error_from_js)?;
        Reflect::set(
            &feeds,
            &JsValue::from_str("scale_factor"),
            &scale_factor_tensor,
        )
        .map_err(error_from_js)?;

        debug!("running browser ONNX inference");
        let outputs = self
            .session
            .run(feeds.into())
            .await
            .map_err(error_from_js)
            .context("run browser ONNX inference")?;

        let mut arrays = Vec::new();
        for key in Reflect::own_keys(&outputs).map_err(error_from_js)?.to_vec() {
            let Some(name) = key.dyn_ref::<JsString>().map(String::from) else {
                continue;
            };
            let value = Reflect::get(&outputs, &key).map_err(error_from_js)?;
            let tensor = value.unchecked_into::<JsTensor>();
            let shape = tensor
                .dims()
                .into_iter()
                .map(|dim| usize::try_from(dim.max(0)).unwrap_or_default())
                .collect::<Vec<_>>();
            let data = match tensor.data() {
                Ok(data) => data,
                Err(_) => tensor.get_data().await.map_err(error_from_js)?,
            };
            let array = Float32Array::new(&data);
            let mut values = vec![0.0; array.length() as usize];
            array.copy_to(&mut values);
            arrays.push((
                name,
                shape.clone(),
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values)
                    .context("convert browser ONNX output to ndarray")?,
            ));
        }
        Ok(arrays)
    }
}

/// Builds ORT Web session options with WebGPU first and WASM as fallback.
fn browser_session_options() -> JsValue {
    let webgpu = Object::new();
    let wasm = Object::new();
    let providers = Array::new();

    let _ = Reflect::set(
        &webgpu,
        &JsValue::from_str("name"),
        &JsValue::from_str("webgpu"),
    );
    let _ = Reflect::set(
        &wasm,
        &JsValue::from_str("name"),
        &JsValue::from_str("wasm"),
    );
    providers.push(&webgpu);
    providers.push(&wasm);

    let options = Object::new();
    let _ = Reflect::set(
        &options,
        &JsValue::from_str("executionProviders"),
        &providers,
    );
    options.into()
}

/// Copies Rust f32 tensor data into a JS-owned ORT Web tensor.
fn js_tensor_from_f32(shape: &[usize], data: &[f32]) -> Result<JsTensor> {
    let buffer = Float32Array::new_with_length(data.len() as u32);
    buffer.copy_from(data);
    JsTensor::new("float32", buffer.into(), js_dims(shape)).map_err(error_from_js)
}

/// Converts Rust tensor dimensions into the JS array shape expected by ORT Web.
fn js_dims(shape: &[usize]) -> JsValue {
    shape
        .iter()
        .map(|dim| JsValue::from_f64(*dim as f64))
        .collect::<Array>()
        .into()
}

/// Loads ORT Web runtime assets and registers the `ort` alternative backend API.
async fn init_ort_web(ort_dist_base_url: Option<String>) -> std::result::Result<(), JsValue> {
    let api = match ort_dist_base_url.filter(|url| !url.trim().is_empty()) {
        Some(base_url) => {
            let dist = ort_web::Dist::new(base_url).with_script_name("ort.webgpu.min.js");
            ort_web::api(dist).await.map_err(js_error)?
        }
        None => ort_web::api(ort_web::FEATURE_WEBGPU)
            .await
            .map_err(js_error)?,
    };

    if !ort::set_api(api) {
        debug!("ort API was already initialized");
    }
    Ok(())
}

/// Configures panic reporting and debug-level tracing for browser console logs.
fn init_browser_diagnostics() {
    console_error_panic_hook::set_once();
    let mut config = tracing_wasm::WASMLayerConfigBuilder::new();
    config.set_max_level(tracing::Level::DEBUG);
    use tracing_subscriber::layer::SubscriberExt;
    let _ = tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(tracing_wasm::WASMLayer::new(config.build())),
    );
}

/// Validates RGBA byte length and builds an owned image buffer.
fn rgba_image_from_bytes(rgba: &[u8], width: u32, height: u32) -> Result<RgbaImage> {
    if width == 0 || height == 0 {
        return Err(Error::InvalidInput {
            message: "input image dimensions must be non-zero".to_string(),
        });
    }

    let expected_len = width as usize * height as usize * 4;
    if rgba.len() != expected_len {
        return Err(Error::InvalidInput {
            message: format!(
                "RGBA input length mismatch: expected {expected_len} bytes for {width}x{height}, got {}",
                rgba.len()
            ),
        });
    }

    RgbaImage::from_raw(width, height, rgba.to_vec()).ok_or_else(|| Error::InvalidInput {
        message: format!("RGBA buffer does not match {width}x{height}"),
    })
}

/// Serializes page detections into a browser-friendly JS value.
fn json_object(page: &WasmPageDetections) -> std::result::Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(page).map_err(js_error)
}

/// Copies annotated RGBA bytes into a JS `Uint8ClampedArray`.
fn rgba_array(rgba: &[u8]) -> std::result::Result<JsValue, JsValue> {
    let image = Uint8ClampedArray::new_with_length(rgba.len() as u32);
    image.copy_from(rgba);
    Ok(image.into())
}

/// Ensures the detection confidence threshold is within the supported range.
fn validate_threshold(threshold: f32) -> Result<()> {
    if (0.0..=1.0).contains(&threshold) {
        return Ok(());
    }

    Err(Error::InvalidInput {
        message: format!("threshold must be in [0, 1], got {threshold}"),
    })
}

/// Formats model output names and shapes for diagnostic errors.
fn format_output_shapes(outputs: &[(String, Vec<usize>, ndarray::ArrayD<f32>)]) -> String {
    outputs
        .iter()
        .map(|(name, shape, _)| format!("{name}: {shape:?}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Converts displayable Rust errors into JavaScript string errors.
fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

/// Wraps JavaScript ORT Web errors in the project error type.
fn error_from_js(error: JsValue) -> Error {
    Error::ModelOutput {
        message: format!("browser ONNX Runtime error: {}", js_value_to_string(&error)),
    }
}

/// Extracts a useful string representation from arbitrary JavaScript error values.
fn js_value_to_string(value: &JsValue) -> String {
    value
        .as_string()
        .or_else(|| {
            js_sys::JSON::stringify(value)
                .ok()
                .and_then(|value| value.as_string())
        })
        .unwrap_or_else(|| format!("{value:?}"))
}

#[allow(dead_code)]
/// Keeps `OriginalSize` referenced in wasm builds where native PDF APIs are disabled.
fn _assert_original_size_is_serializable(_: OriginalSize) {}
