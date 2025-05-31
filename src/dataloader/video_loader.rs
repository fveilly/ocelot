#[cfg(feature = "video")]
use std::path::{Path, PathBuf};
use std::fs;
use image::{DynamicImage, ImageBuffer, Rgb};
use opencv::{core, imgproc, prelude::*, videoio};
use anyhow::Result;

use crate::dataloader::common::{Sample, Compose, TransformConfig, is_video_file};

/// Video Loader for processing video files frame by frame
pub struct VideoLoader {
    videos: Vec<PathBuf>,
    current_video_index: usize,
    current_cap: Option<videoio::VideoCapture>,
    fps: Option<f64>,
    transform: Option<Compose>,
}

impl VideoLoader {
    pub fn new<P: AsRef<Path>>(root: P, transform_config: Option<TransformConfig>) -> Result<Self> {
        let root = root.as_ref();

        let videos = if root.is_dir() {
            fs::read_dir(root)?
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| is_video_file(path))
                .collect()
        } else if root.is_file() && is_video_file(root) {
            vec![root.to_path_buf()]
        } else {
            vec![]
        };

        Ok(Self {
            videos,
            current_video_index: 0,
            current_cap: None,
            fps: None,
            transform: transform_config.map(|_| Compose::new(vec![])), // Placeholder
        })
    }

    pub fn len(&self) -> usize {
        self.videos.len()
    }

    pub fn get_fps(&self) -> Option<f64> {
        self.fps
    }
}

impl Iterator for VideoLoader {
    type Item = Result<Sample>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_video_index >= self.videos.len() {
            return None;
        }

        let result = (|| -> Result<Sample> {
            // Initialize video capture if needed
            if self.current_cap.is_none() {
                let video_path = &self.videos[self.current_video_index];
                let mut cap = videoio::VideoCapture::from_file(
                    &video_path.to_string_lossy(),
                    videoio::CAP_ANY,
                )?;

                self.fps = Some(cap.get(videoio::CAP_PROP_FPS)?);
                self.current_cap = Some(cap);
            }

            let cap = self.current_cap.as_mut().unwrap();
            let mut frame = core::Mat::default();

            let name = self.videos[self.current_video_index]
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            if cap.read(&mut frame)? && !frame.empty() {
                // Convert BGR to RGB
                let mut rgb_frame = core::Mat::default();
                imgproc::cvt_color(&frame, &mut rgb_frame, imgproc::COLOR_BGR2RGB, 0)?;

                // Convert Mat to DynamicImage
                let (height, width) = (rgb_frame.rows() as u32, rgb_frame.cols() as u32);
                let data = rgb_frame.data_bytes()?.to_vec();
                let img_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(width, height, data)
                    .ok_or_else(|| anyhow::anyhow!("Failed to create image buffer"))?;
                let image = DynamicImage::ImageRgb8(img_buffer);

                let shape = (height, width);

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
            } else {
                // End of video, move to next
                self.current_cap = None;
                self.current_video_index += 1;

                // Return empty sample to indicate end of current video
                Ok(Sample {
                    image: DynamicImage::new_rgb8(1, 1),
                    gt: None,
                    mask: None,
                    name,
                    shape: (0, 0),
                    original: None,
                    image_resized: None,
                })
            }
        })();

        Some(result)
    }
}

impl Drop for VideoLoader {
    fn drop(&mut self) {
        if let Some(mut cap) = self.current_cap.take() {
            let _ = cap.release();
        }
    }
}