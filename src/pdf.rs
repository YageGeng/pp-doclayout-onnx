use std::{fs, path::Path};

use image::{ColorType, ImageEncoder, RgbaImage, codecs::png::PngEncoder};
use pdfium::{Document, Library};
use tracing::{debug, info};

use crate::{Error, Result, ResultExt};

/// A single rendered PDF page with an RGBA buffer suitable for annotation.
#[derive(Debug, Clone)]
pub struct RenderedPdfPage {
    pub page_number: u32,
    pub width: u32,
    pub height: u32,
    pub page_width: f32,
    pub page_height: f32,
    pub rgba: RgbaImage,
}

/// Owns a live PDFium library session for loading PDF documents.
pub struct PdfiumSession {
    library: Library,
}

impl Default for PdfiumSession {
    fn default() -> Self {
        Self::new()
    }
}

impl PdfiumSession {
    /// Initializes PDFium and acquires its process-wide serialization lock.
    pub fn new() -> Self {
        Self {
            library: Library::init(),
        }
    }

    /// Reads and loads a PDF document once for repeated page rendering.
    pub fn open_document(&self, pdf_path: impl AsRef<Path>) -> Result<LoadedPdfDocument<'_>> {
        let pdf_path = pdf_path.as_ref();
        info!(path = %pdf_path.display(), "opening PDF document");
        LoadedPdfDocument::open(&self.library, pdf_path)
    }
}

/// Holds one loaded PDF document and the backing bytes required by PDFium.
pub struct LoadedPdfDocument<'lib> {
    document: Document<'lib>,
    _pdf_bytes: Vec<u8>,
}

impl<'lib> LoadedPdfDocument<'lib> {
    /// Reads a PDF file and loads it into PDFium using the provided session.
    pub fn open(library: &'lib Library, pdf_path: impl AsRef<Path>) -> Result<Self> {
        let pdf_path = pdf_path.as_ref();
        let pdf_bytes =
            fs::read(pdf_path).with_context(|| format!("read PDF {}", pdf_path.display()))?;
        let document = library
            .load_document_from_bytes(&pdf_bytes, None)
            .with_context(|| format!("load PDF {}", pdf_path.display()))?;
        debug!(path = %pdf_path.display(), bytes = pdf_bytes.len(), "loaded PDF bytes");

        Ok(Self {
            document,
            _pdf_bytes: pdf_bytes,
        })
    }

    /// Returns the number of pages in the loaded PDF document.
    pub fn page_count(&self) -> Result<usize> {
        usize::try_from(self.document.page_count()).map_err(|error| Error::InvalidInput {
            message: format!("convert PDF page count: {error}"),
        })
    }

    /// Renders one zero-based page from the loaded PDF document.
    pub fn render_page(&self, page_index: usize, dpi: f32) -> Result<RenderedPdfPage> {
        if dpi <= 0.0 {
            return Err(Error::InvalidInput {
                message: format!("dpi must be greater than zero, got {dpi}"),
            });
        }

        let page_number = page_index as u32 + 1;
        let pdfium_page_index = i32::try_from(page_index).map_err(|error| Error::InvalidInput {
            message: format!("convert PDF page index {page_index}: {error}"),
        })?;
        debug!(page_number, dpi, "rendering PDF page");
        let page = self
            .document
            .page(pdfium_page_index)
            .with_context(|| format!("load PDF page {page_number}"))?;
        let page_width = page.width();
        let page_height = page.height();
        let bitmap = page
            .render(dpi)
            .with_context(|| format!("render PDF page {page_number}"))?;
        let width = bitmap.width() as u32;
        let height = bitmap.height() as u32;
        let rgba = bitmap.to_rgba();
        let rgba = RgbaImage::from_raw(width, height, rgba).ok_or_else(|| Error::ModelOutput {
            message: format!(
                "PDF page {page_number} rendered RGBA buffer does not match {width}x{height}"
            ),
        })?;
        debug!(page_number, width, height, "rendered PDF page");

        Ok(RenderedPdfPage {
            page_number,
            width,
            height,
            page_width,
            page_height,
            rgba,
        })
    }

    /// Renders pages one at a time and passes each rendered page to `visitor`.
    pub fn visit_rendered_pages(
        &self,
        dpi: f32,
        mut visitor: impl FnMut(RenderedPdfPage) -> Result<()>,
    ) -> Result<()> {
        let page_count = self.page_count()?;
        info!(page_count, dpi, "rendering PDF pages");
        for page_index in 0..page_count {
            visitor(self.render_page(page_index, dpi)?)?;
        }

        Ok(())
    }
}

/// Builds a stable output file path for a one-based PDF page number.
pub fn output_path_for_page(
    output_dir: impl AsRef<Path>,
    page_number: u32,
    extension: &str,
) -> std::path::PathBuf {
    output_dir
        .as_ref()
        .join(format!("page-{page_number:04}.{extension}"))
}

/// Encodes an RGBA image as PNG bytes.
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
