use image::{DynamicImage, RgbImage, imageops::FilterType};
use ndarray::{Array2, Array4};

use super::{OriginalSize, PreprocessedImage, TARGET_SIZE};
use crate::{Error, Result, ResultExt};

impl TryFrom<&DynamicImage> for PreprocessedImage {
    type Error = Error;

    /// Converts a dynamic image into preprocessed model tensors through RGB normalization.
    fn try_from(image: &DynamicImage) -> Result<Self> {
        Self::try_from(&image.to_rgb8())
    }
}

impl TryFrom<&RgbImage> for PreprocessedImage {
    type Error = Error;

    /// Converts an RGB image into the resized tensors expected by PP-DocLayoutV3.
    fn try_from(image: &RgbImage) -> Result<Self> {
        if image.width() == 0 || image.height() == 0 {
            return Err(Error::InvalidInput {
                message: "input image dimensions must be non-zero".to_string(),
            });
        }

        let original_size = OriginalSize::new(image.width(), image.height());
        let resized =
            image::imageops::resize(image, TARGET_SIZE, TARGET_SIZE, FilterType::Triangle);
        let scale_y = TARGET_SIZE as f32 / image.height() as f32;
        let scale_x = TARGET_SIZE as f32 / image.width() as f32;
        Ok(PreprocessedImage {
            tensor: image_to_nchw_rgb(&resized),
            im_shape: Array2::from_shape_vec((1, 2), vec![TARGET_SIZE as f32, TARGET_SIZE as f32])
                .context("build im_shape tensor")?,
            scale_factor: Array2::from_shape_vec((1, 2), vec![scale_y, scale_x])
                .context("build scale_factor tensor")?,
            original_size,
        })
    }
}

/// Converts an RGB image into a `[1, 3, height, width]` NCHW tensor in `[0, 1]`.
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
