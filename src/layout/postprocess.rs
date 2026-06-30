use ndarray::{Array3, ArrayD, Ix2, Ix3};

use super::{Detection, OriginalSize, PPDocLayoutV3Label, TARGET_SIZE};
use crate::{Error, Result, ResultExt};

/// Parses DETR-style logits and normalized boxes into sorted layout detections.
pub fn parse_detr_outputs(
    logits: ArrayD<f32>,
    boxes: ArrayD<f32>,
    original_size: OriginalSize,
    threshold: f32,
) -> Result<Vec<Detection>> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err(Error::InvalidInput {
            message: format!("threshold must be in [0, 1], got {threshold}"),
        });
    }

    let logits = logits
        .into_dimensionality::<Ix3>()
        .context("logits output should have shape [1, queries, classes]")?;
    let boxes = boxes
        .into_dimensionality::<Ix3>()
        .context("boxes output should have shape [1, queries, 4]")?;

    let logit_shape = logits.shape();
    let box_shape = boxes.shape();
    let (logit_batch, query_count, class_count) = (logit_shape[0], logit_shape[1], logit_shape[2]);
    let (box_batch, box_count, box_dims) = (box_shape[0], box_shape[1], box_shape[2]);

    if logit_batch != 1 || box_batch != 1 {
        return Err(Error::ModelOutput {
            message: format!(
                "only batch size 1 is supported, got logits batch {logit_batch}, boxes batch {box_batch}"
            ),
        });
    }
    if query_count != box_count || box_dims != 4 {
        return Err(Error::ModelOutput {
            message: format!(
                "logits and boxes shape mismatch: logits {:?}, boxes {:?}",
                logits.shape(),
                boxes.shape()
            ),
        });
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
                .with_context(|| format!("convert class id {best_class} to label"))?,
            score,
            bbox,
            order: None,
        });
    }

    detections.sort_by(|left, right| right.score.total_cmp(&left.score));
    Ok(detections)
}

/// Parses Paddle fetch outputs with `[class_id, score, x1, y1, x2, y2, ...]` rows.
pub fn parse_paddle_fetch_output(
    output: ArrayD<f32>,
    original_size: OriginalSize,
    threshold: f32,
) -> Result<Vec<Detection>> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err(Error::InvalidInput {
            message: format!("threshold must be in [0, 1], got {threshold}"),
        });
    }

    let output = output
        .into_dimensionality::<Ix2>()
        .context("Paddle fetch output should have shape [boxes, 6 or 7]")?;
    let columns = output.shape()[1];
    if columns != 6 && columns != 7 {
        return Err(Error::ModelOutput {
            message: format!("Paddle fetch output should have 6 or 7 columns, got {columns}"),
        });
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
            order: parse_reading_order(row.get(6).copied()),
        });
    }

    sort_paddle_detections(&mut detections);
    Ok(detections)
}

/// Converts center-size box coordinates into clamped top-left/bottom-right coordinates.
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

/// Computes the softmax probability for a single query and class.
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

/// Clamps an `xyxy` box to the rendered page bounds.
fn clamp_xyxy(bbox: [f32; 4], original_size: OriginalSize) -> [f32; 4] {
    [
        bbox[0].clamp(0.0, original_size.width as f32),
        bbox[1].clamp(0.0, original_size.height as f32),
        bbox[2].clamp(0.0, original_size.width as f32),
        bbox[3].clamp(0.0, original_size.height as f32),
    ]
}

/// Parses the optional exported reading-order column from Paddle output rows.
fn parse_reading_order(value: Option<f32>) -> Option<usize> {
    let value = value?;
    if !value.is_finite() || value < 0.0 {
        return None;
    }

    Some(value.round() as usize)
}

/// Sorts Paddle detections by reading order when present, then by confidence.
fn sort_paddle_detections(detections: &mut [Detection]) {
    detections.sort_by(|left, right| match (left.order, right.order) {
        (Some(left_order), Some(right_order)) => left_order
            .cmp(&right_order)
            .then_with(|| right.score.total_cmp(&left.score)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => right.score.total_cmp(&left.score),
    });
}
