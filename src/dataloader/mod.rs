use std::collections::HashMap;
use image::DynamicImage;
use anyhow::Result;
use serde::{Deserialize, Serialize};

pub mod image_loader;
pub mod rgb_dataset;

#[cfg(feature = "video")]
pub mod video_loader;

#[cfg(feature = "webcam")]
pub mod webcam_loader;

pub mod refinement_loader;

pub use image_loader::ImageLoader;
pub use rgb_dataset::RgbDataset;
pub use refinement_loader::RefinementLoader;

#[cfg(feature = "video")]
pub use video_loader::VideoLoader;

#[cfg(feature = "webcam")]
pub use webcam_loader::WebcamLoader;

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformConfig {
    pub transforms: HashMap<String, Option<serde_json::Value>>,
}

#[derive(Debug, Clone)]
pub struct Sample {
    pub image: DynamicImage,
    pub gt: Option<DynamicImage>,
    pub mask: Option<DynamicImage>,
    pub name: String,
    pub shape: (u32, u32), // (height, width)
    pub original: Option<DynamicImage>,
    pub image_resized: Option<DynamicImage>,
}

pub trait Transform {
    fn apply(&self, sample: &mut Sample) -> Result<()>;
}

pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self { transforms }
    }

    pub fn apply(&self, sample: &mut Sample) -> Result<()> {
        for transform in &self.transforms {
            transform.apply(sample)?;
        }
        Ok(())
    }
}

pub fn sort_paths(paths: &mut Vec<PathBuf>) {
    paths.sort_by(|a, b| {
        a.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .cmp(&b.file_name().unwrap_or_default().to_string_lossy())
    });
}

pub fn is_image_file(path: &Path) -> bool {
    if let Some(ext) = path.extension() {
        let ext = ext.to_string_lossy().to_lowercase();
        matches!(ext.as_str(), "jpg" | "jpeg" | "png")
    } else {
        false
    }
}

#[cfg(any(feature = "video", feature = "webcam"))]
pub fn is_video_file(path: &Path) -> bool {
    if let Some(ext) = path.extension() {
        let ext = ext.to_string_lossy().to_lowercase();
        matches!(ext.as_str(), "mp4" | "avi" | "mov")
    } else {
        false
    }
}