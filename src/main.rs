use std::path::PathBuf;

use clap::{Parser, Subcommand};
use pp_doclayout_onnx::{
    DEFAULT_OUTPUT_DIR, MODEL_URL, OrtDocLayout, Result, ResultExt, detect_pdf_to_output_dir,
};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Run PaddlePaddle PP-DocLayoutV3 ONNX with ort"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Detect document layout regions in every page of a PDF.
    Detect {
        /// Input PDF path.
        input_pdf: PathBuf,
        /// Path to inference.onnx.
        #[arg(long, default_value = "models/inference.onnx")]
        model: PathBuf,
    },
    /// Dump the first f32 values from model outputs for debugging exporter formats.
    Dump {
        /// Path to inference.onnx.
        #[arg(long, default_value = "models/inference.onnx")]
        model: PathBuf,
        /// Input image path.
        image: PathBuf,
        /// Maximum number of f32 values per output tensor.
        #[arg(long, default_value_t = 28)]
        values: usize,
        /// Optional ONNX Runtime intra-op thread count.
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Print the upstream Hugging Face ONNX model URL.
    ModelUrl,
}

/// Parses CLI arguments and dispatches the selected PP-DocLayout command.
fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();

    match cli.command {
        Command::Detect { input_pdf, model } => {
            let outputs = detect_pdf_to_output_dir(input_pdf, model)?;
            println!("wrote {} pages to {}", outputs.len(), DEFAULT_OUTPUT_DIR);
        }
        Command::Dump {
            model,
            image,
            values,
            threads,
        } => {
            let mut detector = OrtDocLayout::new(model, threads)?;
            let dumps = detector.dump_image_outputs(image, values)?;
            let json =
                serde_json::to_string_pretty(&dumps).context("serialize model output dump")?;
            println!("{json}");
        }
        Command::ModelUrl => {
            println!("{MODEL_URL}");
        }
    }

    Ok(())
}

/// Initializes tracing from `RUST_LOG` while keeping logging optional for CLI users.
fn init_tracing() {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .try_init();
}
