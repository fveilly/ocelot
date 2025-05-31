use anyhow::{Result, anyhow};
use tch::{nn::Module, Tensor, Device, Kind};

pub mod model;
pub mod backbone;
pub mod decoder;
pub mod utils;

pub use model::Ocelot;
pub use backbone::{Res2Net, SwinTransformer};
pub use decoder::{SimpleDecoder, PyramidDecoder};

/// Configuration for InSPyReNet model
#[derive(Debug, Clone)]
pub struct Config {
    pub backbone: BackboneType,
    pub depth: usize,
    pub base_size: (i64, i64),  // (height, width)
    pub device: Device,
    pub pretrained: bool,
}

#[derive(Debug, Clone)]
pub enum BackboneType {
    Res2Net50,
    Res2Net101,
    SwinTiny,
    SwinSmall,
    SwinBase,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            backbone: BackboneType::Res2Net50,
            depth: 50,
            base_size: (320, 320),
            device: Device::cuda_if_available(),
            pretrained: true,
        }
    }
}

/// Image preprocessing utilities
pub struct Preprocessor {
    mean: Vec<f32>,
    std: Vec<f32>,
    size: (i64, i64),
}

impl Preprocessor {
    pub fn new(size: (i64, i64)) -> Self {
        Self {
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
            size,
        }
    }

    pub fn preprocess(&self, image: &image::RgbImage) -> Result<Tensor> {
        let (width, height) = image.dimensions();

        // Convert image to tensor
        let tensor = Tensor::zeros(&[1, 3, self.size.0, self.size.1], (Kind::Float, Device::Cpu));

        // Resize and normalize
        let resized = image::imageops::resize(
            image,
            self.size.1 as u32,
            self.size.0 as u32,
            image::imageops::FilterType::Lanczos3,
        );

        for (y, row) in resized.enumerate_rows().enumerate() {
            for (x, pixel) in row.1.enumerate() {
                let [r, g, b] = pixel.2.0;

                // Normalize to [0, 1] then apply ImageNet normalization
                let r_norm = (r as f32 / 255.0 - self.mean[0]) / self.std[0];
                let g_norm = (g as f32 / 255.0 - self.mean[1]) / self.std[1];
                let b_norm = (b as f32 / 255.0 - self.mean[2]) / self.std[2];

                let _ = tensor.get(0).get(0).get(y as i64).get(x as i64).fill(r_norm as f64);
                let _ = tensor.get(0).get(1).get(y as i64).get(x as i64).fill(g_norm as f64);
                let _ = tensor.get(0).get(2).get(y as i64).get(x as i64).fill(b_norm as f64);
            }
        }

        Ok(tensor)
    }
}

/// Post-processing utilities for pyramid blending
pub struct Postprocessor {
    pub scales: Vec<f64>,
}

impl Postprocessor {
    pub fn new() -> Self {
        Self {
            scales: vec![0.5, 0.75, 1.0, 1.25, 1.5],
        }
    }

    /// Apply pyramid blending for high-resolution prediction
    pub fn pyramid_blend(&self, predictions: Vec<Tensor>, original_size: (i64, i64)) -> Result<Tensor> {
        if predictions.is_empty() {
            return Err(anyhow!("No predictions provided"));
        }

        let mut blended = Tensor::zeros(&[1, 1, original_size.0, original_size.1],
                                        (Kind::Float, predictions[0].device()));

        let weights: Vec<f32> = vec![0.1, 0.2, 0.4, 0.2, 0.1]; // Weights for different scales

        for (i, pred) in predictions.iter().enumerate() {
            let weight = weights.get(i).unwrap_or(&0.2);

            // Resize prediction to original size
            let resized = pred.upsample_bilinear2d(
                &[original_size.0, original_size.1],
                false,
                None,
                None,
            );

            blended = blended + resized * *weight as f64;
        }

        Ok(blended)
    }

    /// Convert tensor to image
    pub fn tensor_to_image(&self, tensor: &Tensor) -> Result<image::GrayImage> {
        let tensor = tensor.squeeze_dim(0).squeeze_dim(0);
        let (h, w) = (tensor.size2()?.0, tensor.size2()?.1);

        let mut img = image::GrayImage::new(w as u32, h as u32);

        for y in 0..h {
            for x in 0..w {
                let val = f32::try_from(tensor.get(y).get(x))?;
                let pixel_val = (val.clamp(0.0, 1.0) * 255.0) as u8;
                img.put_pixel(x as u32, y as u32, image::Luma([pixel_val]));
            }
        }

        Ok(img)
    }
}

/// Utility functions
pub mod ops {
    use tch::{Tensor, nn};

    pub fn conv2d(vs: &nn::Path, in_channels: i64, out_channels: i64,
                  kernel_size: i64, stride: i64, padding: i64) -> nn::Conv2D {
        nn::conv2d(vs, in_channels, out_channels, kernel_size,
                   nn::ConvConfig { stride, padding, ..Default::default() })
    }

    pub fn batch_norm2d(vs: &nn::Path, num_features: i64) -> nn::BatchNorm {
        nn::batch_norm2d(vs, num_features, Default::default())
    }

    pub fn adaptive_avg_pool2d(input: &Tensor, output_size: (i64, i64)) -> Tensor {
        input.adaptive_avg_pool2d(&[output_size.0, output_size.1])
    }

    pub fn interpolate_bilinear(input: &Tensor, size: (i64, i64)) -> Tensor {
        input.upsample_bilinear2d(&[size.0, size.1], false, None, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.base_size, (320, 320));
        assert_eq!(config.depth, 50);
    }

    #[test]
    fn test_preprocessor() {
        let preprocessor = Preprocessor::new((224, 224));
        assert_eq!(preprocessor.size, (224, 224));
        assert_eq!(preprocessor.mean.len(), 3);
        assert_eq!(preprocessor.std.len(), 3);
    }
}