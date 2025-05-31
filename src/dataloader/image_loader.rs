use std::path::{Path, PathBuf};
use std::fs;
use anyhow::Result;
use crate::dataloader::{is_image_file, sort_paths, Compose, Sample, TransformConfig};

/// Image Loader for inference on individual images or image directories
pub struct ImageLoader {
    images: Vec<PathBuf>,
    current_index: usize,
    transform: Option<Compose>,
}

impl ImageLoader {
    pub fn new<P: AsRef<Path>>(root: P, transform_config: Option<TransformConfig>) -> Result<Self> {
        let root = root.as_ref();

        let images = if root.is_dir() {
            let mut imgs: Vec<PathBuf> = fs::read_dir(root)?
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| is_image_file(path))
                .collect();
            sort_paths(&mut imgs);
            imgs
        } else if root.is_file() && is_image_file(root) {
            vec![root.to_path_buf()]
        } else {
            vec![]
        };

        Ok(Self {
            images,
            current_index: 0,
            transform: transform_config.map(|_| Compose::new(vec![])), // Placeholder
        })
    }

    pub fn len(&self) -> usize {
        self.images.len()
    }

    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    pub fn reset(&mut self) {
        self.current_index = 0;
    }
}

impl Iterator for ImageLoader {
    type Item = Result<Sample>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.images.len() {
            return None;
        }

        let result = (|| -> Result<Sample> {
            let image = image::open(&self.images[self.current_index])?;
            let shape = (image.height(), image.width());
            let name = self.images[self.current_index]
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            let mut sample = Sample {
                image: image.clone(),
                gt: None,
                mask: None,
                name,
                shape,
                original: Some(image),
                image_resized: None,
            };

            if let Some(transform) = &self.transform {
                transform.apply(&mut sample)?;
            }

            Ok(sample)
        })();

        self.current_index += 1;
        Some(result)
    }
}