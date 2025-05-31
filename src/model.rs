use tch::{nn, nn::Module, Tensor, Device, Kind, Reduction, TchError};
use anyhow::{anyhow, Result};
use crate::{Config, BackboneType, backbone::*, decoder::*};

pub struct Ocelot {
    pub vs: nn::VarStore,
    backbone: Box<dyn BackboneModule>,
    decoder: Box<dyn DecoderModule>,
    pub(crate) config: Config,
}

impl Ocelot {
    pub fn new(vs: &nn::Path, config: Config) -> Result<Self> {
        let backbone: Box<dyn BackboneModule> = match config.backbone {
            BackboneType::Res2Net50 => Box::new(Res2Net::new(&(vs / "backbone"), 50, config.pretrained)?),
            BackboneType::Res2Net101 => Box::new(Res2Net::new(&(vs / "backbone"), 101, config.pretrained)?),
            BackboneType::SwinTiny => Box::new(SwinTransformer::new(&(vs / "backbone"), "tiny", config.pretrained)?),
            BackboneType::SwinSmall => Box::new(SwinTransformer::new(&(vs / "backbone"), "small", config.pretrained)?),
            BackboneType::SwinBase => Box::new(SwinTransformer::new(&(vs / "backbone"), "base", config.pretrained)?),
        };

        let decoder = Box::new(PyramidDecoder::new(&(vs / "decoder"), &config)?);

        Ok(Self {
            vs: nn::VarStore::new(config.device),
            backbone,
            decoder,
            config,
        })
    }
    
    #[inline]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Forward pass for multi-scale prediction
    pub fn forward_ms(&self, x: &Tensor, scales: &[f64], train: bool) -> Result<Vec<Tensor>> {
        let mut results = Vec::new();
        let original_size = (x.size()[2], x.size()[3]);

        for &scale in scales {
            let scaled_size = (
                (original_size.0 as f64 * scale) as i64,
                (original_size.1 as f64 * scale) as i64,
            );

            // Resize input
            let scaled_input = x.upsample_bilinear2d(
                &[scaled_size.0, scaled_size.1],
                false,
                None,
                None,
            );

            // Forward pass
            let features = self.backbone.forward_t(&scaled_input, train)?;
            let prediction = self.decoder.forward_t(&features, train)?;

            // Resize back to original size
            let resized_pred = prediction.upsample_bilinear2d(
                &[original_size.0, original_size.1],
                false,
                None,
                None,
            );

            results.push(resized_pred);
        }

        Ok(results)
    }

    /// Single scale forward pass
    pub fn forward_t(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let features = self.backbone.forward_t(x, train)?;
        self.decoder.forward_t(&features, train)
    }

    /// Pyramid blending for high-resolution inference
    pub fn predict_pyramid(&self, x: &Tensor) -> Result<Tensor> {
        let scales = vec![0.5, 0.75, 1.0, 1.25, 1.5];
        let predictions = self.forward_ms(x, &scales, false)?;

        // Blend predictions with learned weights
        self.blend_predictions(predictions)
    }

    fn blend_predictions(&self, predictions: Vec<Tensor>) -> Result<Tensor> {
        if predictions.is_empty() {
            return Err(anyhow::anyhow!("No predictions to blend"));
        }

        // Gaussian weights for pyramid blending
        let weights = vec![0.06, 0.24, 0.4, 0.24, 0.06];
        let mut result = Tensor::zeros_like(&predictions[0]);

        for (pred, &weight) in predictions.iter().zip(weights.iter()) {
            result = result + pred * weight;
        }

        Ok(result)
    }

    /// Load pretrained weights
    pub fn load_weights(&mut self, path: &str) -> Result<(), TchError> {
        self.vs.load(path)
    }

    /// Save current weights
    pub fn save_weights(&self, path: &str) -> Result<(), TchError> {
        self.vs.save(path)
    }
}

impl Ocelot {
    pub fn compute_loss(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        // Multi-scale supervision loss
        let bce_loss = predictions.binary_cross_entropy_with_logits::<Tensor>(targets, None, None, Reduction::Mean);

        // IoU loss component
        let pred_sigmoid = predictions.sigmoid();
        let intersection = (&pred_sigmoid * targets).sum_dim_intlist(&[2i64, 3i64][..], false, Kind::Float);
        let union = pred_sigmoid.sum_dim_intlist(&[2i64, 3i64][..], false, Kind::Float) +
            targets.sum_dim_intlist(&[2i64, 3i64][..], false, Kind::Float) -
            &intersection;

        let iou = intersection / (union + 1e-8);
        let iou_loss = ((1.0 - iou) as Tensor).mean(Kind::Float);

        // Combined loss
        bce_loss + iou_loss * 0.5
    }

    /// Training step
    pub fn train_step(&self, x: &Tensor, targets: &Tensor, opt: &mut nn::Optimizer) -> Result<f64> {
        let predictions = self.forward_t(x, true)?;
        let loss = self.compute_loss(&predictions, targets);

        opt.zero_grad();
        loss.backward();
        opt.step();

        f64::try_from(loss).map_err(|e| anyhow!(e))
    }
}

impl Ocelot {
    /// Predict on single image with TTA (Test Time Augmentation)
    pub fn predict_with_tta(&self, x: &Tensor) -> Result<Tensor> {
        let mut predictions = Vec::new();

        // Original
        predictions.push(self.predict_pyramid(x)?);

        // Horizontal flip
        let x_hflip = x.flip(&[3]);
        let pred_hflip = self.predict_pyramid(&x_hflip)?.flip(&[3]);
        predictions.push(pred_hflip);

        // Vertical flip
        let x_vflip = x.flip(&[2]);
        let pred_vflip = self.predict_pyramid(&x_vflip)?.flip(&[2]);
        predictions.push(pred_vflip);

        // Average all predictions
        let mut result = Tensor::zeros_like(&predictions[0]);
        for pred in &predictions {
            result = result + pred;
        }
        result = result / predictions.len() as f64;

        Ok(result)
    }

    /// Batch inference
    pub fn predict_batch(&self, batch: &Tensor) -> Result<Tensor> {
        let batch_size = batch.size()[0];
        let mut results = Vec::new();

        for i in 0..batch_size {
            let single_image = batch.get(i).unsqueeze(0);
            let prediction = self.predict_pyramid(&single_image)?;
            results.push(prediction);
        }

        Ok(Tensor::cat(&results, 0))
    }
}

/// Model builder pattern
pub struct InSPyReNetBuilder {
    config: Config,
}

impl InSPyReNetBuilder {
    pub fn new() -> Self {
        Self {
            config: Config::default(),
        }
    }

    pub fn backbone(mut self, backbone: BackboneType) -> Self {
        self.config.backbone = backbone;
        self
    }

    pub fn base_size(mut self, size: (i64, i64)) -> Self {
        self.config.base_size = size;
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.config.device = device;
        self
    }

    pub fn pretrained(mut self, pretrained: bool) -> Self {
        self.config.pretrained = pretrained;
        self
    }

    pub fn build(self, vs: &nn::Path) -> Result<Ocelot> {
        Ocelot::new(vs, self.config)
    }
}

impl Default for InSPyReNetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind};

    #[test]
    fn test_model_creation() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let config = Config {
            backbone: BackboneType::Res2Net50,
            base_size: (320, 320),
            device,
            ..Default::default()
        };

        // This would require implementing the actual backbone and decoder
        // let model = InSPyReNet::new(&vs.root(), config);
        // assert!(model.is_ok());
    }

    #[test]
    fn test_builder_pattern() {
        let builder = InSPyReNetBuilder::new()
            .backbone(BackboneType::Res2Net50)
            .base_size((416, 416))
            .device(Device::Cpu)
            .pretrained(true);

        assert_eq!(builder.config.base_size, (416, 416));
    }
}