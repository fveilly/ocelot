use std::fs;
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};
use crate::dataloader::{is_image_file, sort_paths, Compose, Sample, TransformConfig};

pub struct RgbDataset {
    images: Vec<PathBuf>,
    gts: Vec<PathBuf>,
    transform: Option<Compose>,
}

impl RgbDataset {
    pub fn new<P: AsRef<Path>>(
        root: P,
        sets: Vec<&str>,
        transform_config: Option<TransformConfig>,
    ) -> Result<Self> {
        let mut images = Vec::new();
        let mut gts = Vec::new();

        for set in sets {
            let image_root = root.as_ref().join(set).join("images");
            let gt_root = root.as_ref().join(set).join("masks");

            if !image_root.exists() || !gt_root.exists() {
                continue;
            }

            let mut set_images: Vec<PathBuf> = fs::read_dir(&image_root)?
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| is_image_file(path))
                .collect();
            sort_paths(&mut set_images);

            let mut set_gts: Vec<PathBuf> = fs::read_dir(&gt_root)?
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| is_image_file(path))
                .collect();
            sort_paths(&mut set_gts);

            images.extend(set_images);
            gts.extend(set_gts);
        }

        let mut dataset = Self {
            images,
            gts,
            transform: transform_config.map(|_| Compose::new(vec![])), // Placeholder for actual transforms
        };

        dataset.filter_files()?;
        Ok(dataset)
    }

    fn filter_files(&mut self) -> Result<()> {
        let mut filtered_images = Vec::new();
        let mut filtered_gts = Vec::new();

        for (img_path, gt_path) in self.images.iter().zip(self.gts.iter()) {
            let img = image::open(img_path)
                .with_context(|| format!("Failed to open image: {:?}", img_path))?;
            let gt = image::open(gt_path)
                .with_context(|| format!("Failed to open ground truth: {:?}", gt_path))?;

            if img.dimensions() == gt.dimensions() {
                filtered_images.push(img_path.clone());
                filtered_gts.push(gt_path.clone());
            }
        }

        self.images = filtered_images;
        self.gts = filtered_gts;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.images.len()
    }

    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    pub fn get_item(&self, index: usize) -> Result<Sample> {
        if index >= self.len() {
            return Err(anyhow::anyhow!("Index out of bounds"));
        }

        let image = image::open(&self.images[index])?;
        let gt = image::open(&self.gts[index])?.to_luma8();
        let shape = (gt.height(), gt.width());
        let name = self.images[index]
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        let mut sample = Sample {
            image: image.clone(),
            gt: Some(DynamicImage::ImageLuma8(gt)),
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
    }
}
