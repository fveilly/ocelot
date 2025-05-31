#[cfg(feature = "webcam")]
use image::{DynamicImage, ImageBuffer, Rgb};
use opencv::{core, imgproc, prelude::*, videoio};
use anyhow::Result;

use crate::dataloader::common::{Sample, Compose, TransformConfig};

/// Webcam Loader for real-time webcam input
pub struct WebcamLoader {
    cap: videoio::VideoCapture,
    transform: Option<Compose>,
}

impl WebcamLoader {
    pub fn new(device_id: i32, transform_config: Option<TransformConfig>) -> Result<Self> {
        let mut cap = videoio::VideoCapture::new(device_id, videoio::CAP_ANY)?;
        cap.set(videoio::CAP_PROP_FRAME_WIDTH, 640.0)?;
        cap.set(videoio::CAP_PROP_FRAME_HEIGHT, 480.0)?;

        Ok(Self {
            cap,
            transform: transform_config.map(|_| Compose::new(vec![])), // Placeholder
        })
    }

    pub fn set_resolution(&mut self, width: f64, height: f64) -> Result<()> {
        self.cap.set(videoio::CAP_PROP_FRAME_WIDTH, width)?;
        self.cap.set(videoio::CAP_PROP_FRAME_HEIGHT, height)?;
        Ok(())
    }
}

impl Iterator for WebcamLoader {
    type Item = Result<Sample>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = (|| -> Result<Sample> {
            let mut frame = core::Mat::default();

            if self.cap.read(&mut frame)? && !frame.empty() {
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
                    name: "webcam".to_string(),
                    shape,
                    original: Some(image),
                    image_resized: None,
                };

                if let Some(transform) = &self.transform {
                    transform.apply(&mut sample)?;
                }

                Ok(sample)
            } else {
                Err(anyhow::anyhow!("Failed to read from webcam"))
            }
        })();

        Some(result)
    }
}

impl Drop for WebcamLoader {
    fn drop(&mut self) {
        let _ = self.cap.release();
    }
}