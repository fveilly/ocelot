use std::path::{Path, PathBuf};
use std::fs;
use image::DynamicImage;
use anyhow::Result;
use crate::dataloader::{is_image_file, sort_paths, Compose, Sample, TransformConfig};

/// Refinement Loader for refinement tasks with image-segmentation pairs
pub struct RefinementLoader {
    images: Vec<PathBuf>,
    segs: Vec<PathBuf>,
    current_index: usize,
    transform: Option<Compose>,
}

impl RefinementLoader {
    pub fn new<P: AsRef<Path>>(
        image_dir: P,
        seg_dir: P,
        transform_config: Option<TransformConfig>,
    ) -> Result<Self> {
        let mut images: Vec<PathBuf> = fs::read_dir(image_dir)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| is_image_file(path))
            .collect();
        sort_paths(&mut images);

        let mut segs: Vec<PathBuf> = fs::read_dir(seg_dir)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| is_image_file(path))
            .collect();
        sort_paths(&mut segs);

        Ok(Self {
            images,
            segs,
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

impl Iterator for RefinementLoader {
    type Item = Result<Sample>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.images.len() {
            return None;
        }

        let result = (|| -> Result<Sample> {
            let image = image::open(&self.images[self.current_index])?;
            let seg = image::open(&self.segs[self.current_index])?.to_luma8();
            let shape = (image.height(), image.width());
            let name = self.images[self.current_index]
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            let mut sample = Sample {
                image: image.clone(),
                gt: Some(DynamicImage::ImageLuma8(seg.clone())),
                mask: Some(DynamicImage::ImageLuma8(seg)),
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