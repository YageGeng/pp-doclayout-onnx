use std::path::PathBuf;

use ort::session::builder::SessionBuilder;
#[cfg(feature = "native")]
use pdfium::PdfiumError;
use thiserror::Error as ThisError;

use crate::PPDocLayoutV3LabelError;

/// Project-wide result type used by all fallible PP-DocLayout operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Project-wide error type with explicit variants for IO, model, PDF, and tensor failures.
#[derive(Debug, ThisError)]
pub enum Error {
    #[error("{message}")]
    InvalidInput { message: String },
    #[error(
        "model file does not exist: {path}\ndownload it with:\n  uv --cache-dir models/.uv-cache run python scripts/download_model.py"
    )]
    MissingModel { path: PathBuf },
    #[error("{message}")]
    Context {
        message: String,
        #[source]
        source: Box<Error>,
    },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Image(#[from] image::ImageError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Shape(#[from] ndarray::ShapeError),
    #[cfg(feature = "native")]
    #[error(transparent)]
    Pdfium(#[from] PdfiumError),
    #[error(transparent)]
    Ort(#[from] ort::Error),
    #[error(transparent)]
    OrtBuilder(Box<ort::Error<SessionBuilder>>),
    #[error(transparent)]
    Label(#[from] PPDocLayoutV3LabelError),
    #[error("{message}")]
    ModelOutput { message: String },
}

/// Adds context to fallible operations while preserving the original project error as source.
pub trait ResultExt<T> {
    /// Wraps the error with a static or prebuilt context message.
    fn context(self, message: impl Into<String>) -> Result<T>;

    /// Wraps the error with a lazily built context message.
    fn with_context(self, message: impl FnOnce() -> String) -> Result<T>;
}

impl<T, E> ResultExt<T> for std::result::Result<T, E>
where
    E: Into<Error>,
{
    fn context(self, message: impl Into<String>) -> Result<T> {
        self.map_err(|source| Error::Context {
            message: message.into(),
            source: Box::new(source.into()),
        })
    }

    fn with_context(self, message: impl FnOnce() -> String) -> Result<T> {
        self.map_err(|source| Error::Context {
            message: message(),
            source: Box::new(source.into()),
        })
    }
}

impl From<ort::Error<SessionBuilder>> for Error {
    /// Converts recoverable ONNX Runtime builder errors into a boxed project error variant.
    fn from(source: ort::Error<SessionBuilder>) -> Self {
        Self::OrtBuilder(Box::new(source))
    }
}
