use image::{Rgb, RgbImage};
use ndarray::{Array2, Array3};
use pp_doclayout_onnx::{
    cxcywh_to_xyxy, image_to_nchw_rgb, output_path_for_page, parse_detr_outputs,
    parse_paddle_fetch_output, OriginalSize, PPDocLayoutV3Label, PdfPageDetections, DEFAULT_DPI,
    DEFAULT_THRESHOLD, PP_DOCLAYOUT_V3_LABELS,
};

#[test]
fn labels_match_pp_doclayout_v3_config() {
    assert_eq!(PPDocLayoutV3Label::class_count(), 25);
    assert_eq!(PP_DOCLAYOUT_V3_LABELS[0], PPDocLayoutV3Label::Abstract);
    assert_eq!(PP_DOCLAYOUT_V3_LABELS[21], PPDocLayoutV3Label::Table);
    assert_eq!(PP_DOCLAYOUT_V3_LABELS[22], PPDocLayoutV3Label::Text);
    assert_eq!(PPDocLayoutV3Label::Table.as_str(), "table");
    assert_eq!(
        "text".parse::<PPDocLayoutV3Label>().unwrap(),
        PPDocLayoutV3Label::Text
    );
    assert_eq!(
        PPDocLayoutV3Label::try_from(21).unwrap(),
        PPDocLayoutV3Label::Table
    );
}

#[test]
fn pdf_detection_defaults_match_model_config() {
    assert_eq!(DEFAULT_DPI, 96.0);
    assert_eq!(DEFAULT_THRESHOLD, 0.5);
}

#[test]
fn ort_dependency_enables_webgpu_feature() {
    let manifest = include_str!("../Cargo.toml");

    assert!(manifest.contains("\"webgpu\""));
}

#[test]
fn output_path_uses_padded_one_based_page_number() {
    assert_eq!(
        output_path_for_page("output", 7, "json"),
        std::path::Path::new("output/page-0007.json")
    );
}

#[test]
fn pdf_page_detection_output_keeps_page_metadata() {
    let page = PdfPageDetections {
        page_number: 3,
        width: 816,
        height: 1056,
        page_width: 612.0,
        page_height: 792.0,
        dpi: DEFAULT_DPI,
        detections: Vec::new(),
    };

    assert_eq!(page.page_number, 3);
    assert_eq!(page.width, 816);
    assert_eq!(page.page_width, 612.0);
    assert_eq!(page.dpi, DEFAULT_DPI);
}

#[test]
fn image_to_nchw_rgb_scales_channels_to_unit_range() {
    let mut image = RgbImage::new(2, 2);
    image.put_pixel(0, 0, Rgb([10, 20, 30]));
    image.put_pixel(1, 0, Rgb([40, 50, 60]));
    image.put_pixel(0, 1, Rgb([70, 80, 90]));
    image.put_pixel(1, 1, Rgb([100, 110, 120]));

    let tensor = image_to_nchw_rgb(&image);

    assert_eq!(tensor.shape(), &[1, 3, 2, 2]);
    assert_eq!(tensor[[0, 0, 0, 0]], 10.0 / 255.0);
    assert_eq!(tensor[[0, 0, 0, 1]], 40.0 / 255.0);
    assert_eq!(tensor[[0, 1, 1, 0]], 80.0 / 255.0);
    assert_eq!(tensor[[0, 2, 1, 1]], 120.0 / 255.0);
}

#[test]
fn normalized_cxcywh_boxes_restore_original_page_coordinates() {
    let bbox = cxcywh_to_xyxy([0.5, 0.5, 0.5, 0.25], OriginalSize::new(200, 100));

    assert_eq!(bbox, [50.0, 37.5, 150.0, 62.5]);
}

#[test]
fn parse_detr_outputs_drops_background_and_keeps_best_classes() {
    let mut logits = Array3::<f32>::from_elem((1, 2, PPDocLayoutV3Label::class_count() + 1), -10.0);
    logits[[0, 0, 21]] = 10.0;
    logits[[0, 1, PPDocLayoutV3Label::class_count()]] = 10.0;
    let boxes = Array2::from_shape_vec(
        (2, 4),
        vec![
            0.5, 0.5, 0.5, 0.25, //
            0.2, 0.2, 0.1, 0.1,
        ],
    )
    .unwrap()
    .into_shape_with_order((1, 2, 4))
    .unwrap();

    let detections = parse_detr_outputs(
        logits.into_dyn(),
        boxes.into_dyn(),
        OriginalSize::new(200, 100),
        0.5,
    )
    .unwrap();

    assert_eq!(detections.len(), 1);
    assert_eq!(detections[0].label, PPDocLayoutV3Label::Table);
    assert_eq!(detections[0].class_id, 21);
    assert!(detections[0].score > 0.99);
    assert_eq!(detections[0].bbox, [50.0, 37.5, 150.0, 62.5]);
    assert_eq!(detections[0].order, None);
}

#[test]
fn parse_paddle_fetch_output_uses_exported_seven_column_boxes_and_reading_order() {
    let output = Array2::from_shape_vec(
        (3, 7),
        vec![
            21.0, 0.90, 10.0, 20.0, 110.0, 120.0, 299.0, //
            22.0, 0.95, 30.0, 40.0, 130.0, 140.0, 298.0, //
            22.0, 0.10, 50.0, 60.0, 150.0, 160.0, 297.0,
        ],
    )
    .unwrap();

    let detections =
        parse_paddle_fetch_output(output.into_dyn(), OriginalSize::new(200, 200), 0.5).unwrap();

    assert_eq!(detections.len(), 2);
    assert_eq!(detections[0].label, PPDocLayoutV3Label::Text);
    assert_eq!(detections[0].class_id, 22);
    assert_eq!(detections[0].score, 0.95);
    assert_eq!(detections[0].bbox, [30.0, 40.0, 130.0, 140.0]);
    assert_eq!(detections[0].order, Some(298));
    assert_eq!(detections[1].label, PPDocLayoutV3Label::Table);
    assert_eq!(detections[1].class_id, 21);
    assert_eq!(detections[1].score, 0.90);
    assert_eq!(detections[1].bbox, [10.0, 20.0, 110.0, 120.0]);
    assert_eq!(detections[1].order, Some(299));
}
