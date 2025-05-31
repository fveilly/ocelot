use tch::{nn, Tensor, Device, Kind, TchError};
use anyhow::Result;
use std::path::Path;
use tch::nn::OptimizerConfig;
use crate::InSPyReNet;

/// Training utilities
pub struct Trainer {
    pub model: InSPyReNet,
    pub optimizer: nn::Optimizer,
    pub device: Device,
    pub config: TrainingConfig,
}

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub batch_size: i64,
    pub epochs: i64,
    pub save_interval: i64,
    pub log_interval: i64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            weight_decay: 5e-4,
            batch_size: 8,
            epochs: 100,
            save_interval: 10,
            log_interval: 100,
        }
    }
}

impl Trainer {
    pub fn new(model: crate::InSPyReNet, config: TrainingConfig) -> Result<Self> {
        let vs = nn::VarStore::new(model.config.device);
        let optimizer = nn::Adam::default().build(&vs, config.learning_rate)?;
        let device = model.config.device;

        Ok(Self {
            model,
            optimizer,
            device,
            config,
        })
    }

    /// Training loop
    pub fn train<D>(&mut self, dataloader: D, val_dataloader: Option<D>) -> Result<()>
    where
        D: Iterator<Item = (Tensor, Tensor)> + Clone,
    {
        for epoch in 0..self.config.epochs {
            let train_loss = self.train_epoch(dataloader.clone())?;

            println!("Epoch {}/{}: Train Loss = {:.6}",
                     epoch + 1, self.config.epochs, train_loss);

            // Validation
            if let Some(val_data) = val_dataloader.clone() {
                let val_loss = self.validate(val_data)?;
                println!("Epoch {}/{}: Val Loss = {:.6}",
                         epoch + 1, self.config.epochs, val_loss);
            }

            // Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0 {
                self.save_checkpoint(&format!("checkpoint_epoch_{}.pt", epoch + 1))?;
            }
        }

        Ok(())
    }

    fn train_epoch<D>(&mut self, dataloader: D) -> Result<f64>
    where
        D: Iterator<Item = (Tensor, Tensor)>,
    {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for (i, (images, targets)) in dataloader.enumerate() {
            let images = images.to_device(self.device);
            let targets = targets.to_device(self.device);

            let loss = self.model.train_step(&images, &targets, &mut self.optimizer)?;
            total_loss += loss;
            num_batches += 1;

            if (i + 1) % self.config.log_interval as usize == 0 {
                println!("Batch {}: Loss = {:.6}", i + 1, loss);
            }
        }

        Ok(total_loss / num_batches as f64)
    }

    fn validate<D>(&self, dataloader: D) -> Result<f64>
    where
        D: Iterator<Item = (Tensor, Tensor)>,
    {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        tch::no_grad(|| -> Result<()> {
            for (images, targets) in dataloader {
                let images = images.to_device(self.device);
                let targets = targets.to_device(self.device);

                let predictions = self.model.forward_t(&images, true)?;
                let loss = self.model.compute_loss(&predictions, &targets);

                total_loss += f64::try_from(loss)?;
                num_batches += 1;
            }
            Ok(())
        })?;

        Ok(total_loss / num_batches as f64)
    }

    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.model.vs.save(path)?;
        Ok(())
    }

    pub fn load_checkpoint<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.model.vs.load(path)?;
        Ok(())
    }
}

/// Loss functions for saliency detection
pub struct SaliencyLoss;

impl SaliencyLoss {
    /// Binary Cross Entropy Loss with logits
    pub fn bce_loss(pred: &Tensor, target: &Tensor) -> Tensor {
        pred.binary_cross_entropy_with_logits::<Tensor>(target, None, None, tch::Reduction::Mean)
    }

    /// IoU Loss (Intersection over Union)
    pub fn iou_loss(pred: &Tensor, target: &Tensor) -> Tensor {
        let pred_sigmoid = pred.sigmoid();
        let intersection = (&pred_sigmoid * target).sum_dim_intlist([2, 3].as_slice(), false, Kind::Float);
        let union = pred_sigmoid.sum_dim_intlist([2, 3].as_slice(), false, Kind::Float)
            + target.sum_dim_intlist([2, 3].as_slice(), false, Kind::Float)
            - &intersection;

        let iou = &intersection / (&union + 1e-8);
        ((1.0 - iou) as Tensor).mean(Kind::Float)
    }

    /// Structure Similarity Loss
    pub fn ssim_loss(pred: &Tensor, target: &Tensor) -> Tensor {
        let pred_sigmoid = pred.sigmoid();

        // Mean
        let mu1 = pred_sigmoid.mean_dim([2, 3].as_slice(), true, Kind::Float);
        let mu2 = target.mean_dim([2, 3].as_slice(), true, Kind::Float);

        // Variance
        let sigma1_sq = ((&pred_sigmoid - &mu1).pow_tensor_scalar(2)).mean_dim([2, 3].as_slice(), true, Kind::Float);
        let sigma2_sq = ((target - &mu2).pow_tensor_scalar(2)).mean_dim([2, 3].as_slice(), true, Kind::Float);

        // Covariance
        let sigma12 = ((&pred_sigmoid - &mu1) * (target - &mu2)).mean_dim([2, 3].as_slice(), true, Kind::Float);

        let c1 = 0.01_f64.powi(2);
        let c2 = 0.03_f64.powi(2);

        let numerator = (2.0 * &mu1 * &mu2 + c1) * (2.0 * &sigma12 + c2);
        let denominator = (&mu1.pow_tensor_scalar(2) + &mu2.pow_tensor_scalar(2) + c1)
            * (&sigma1_sq + &sigma2_sq + c2);

        let ssim = &numerator / &denominator;
        ((1.0 - ssim) as Tensor).mean(Kind::Float)
    }

    /// Combined loss for saliency detection
    pub fn combined_loss(pred: &Tensor, target: &Tensor) -> Tensor {
        let bce = Self::bce_loss(pred, target);
        let iou = Self::iou_loss(pred, target);
        let ssim = Self::ssim_loss(pred, target);

        bce + iou + ssim
    }
}

/// Image preprocessing utilities
pub struct ImageProcessor;

impl ImageProcessor {
    /// Normalize image tensor using ImageNet statistics
    pub fn normalize(image: &Tensor) -> Tensor {
        let mean = Tensor::from_slice(&[0.485, 0.456, 0.406]).view([3, 1, 1]).to_device(image.device());
        let std = Tensor::from_slice(&[0.229, 0.224, 0.225]).view([3, 1, 1]).to_device(image.device());

        (image - mean) / std
    }

    /// Denormalize image tensor
    pub fn denormalize(image: &Tensor) -> Tensor {
        let mean = Tensor::from_slice(&[0.485, 0.456, 0.406]).view([3, 1, 1]).to_device(image.device());
        let std = Tensor::from_slice(&[0.229, 0.224, 0.225]).view([3, 1, 1]).to_device(image.device());

        image * std + mean
    }

    /// Resize image maintaining aspect ratio
    pub fn resize_maintain_ratio(image: &Tensor, target_size: i64) -> Tensor {
        let (_batch, _channels, h, w) = image.size4().unwrap();

        let scale = if h > w {
            target_size as f64 / h as f64
        } else {
            target_size as f64 / w as f64
        };

        let new_h = (h as f64 * scale) as i64;
        let new_w = (w as f64 * scale) as i64;

        image.upsample_bilinear2d(&[new_h, new_w], false, None, None)
    }

    /// Create image pyramid for multi-scale processing
    pub fn create_pyramid(image: &Tensor, scales: &[f64]) -> Vec<Tensor> {
        let (_batch, _channels, h, w) = image.size4().unwrap();

        scales.iter().map(|&scale| {
            let new_h = (h as f64 * scale) as i64;
            let new_w = (w as f64 * scale) as i64;
            image.upsample_bilinear2d(&[new_h, new_w], false, None, None)
        }).collect()
    }

    /// Blend pyramid predictions
    pub fn blend_pyramid(predictions: &[Tensor], target_size: (i64, i64)) -> Tensor {
        let (target_h, target_w) = target_size;

        let mut blended = Tensor::zeros(&[predictions[0].size()[0], 1, target_h, target_w],
                                        (Kind::Float, predictions[0].device()));

        for pred in predictions {
            let resized = pred.upsample_bilinear2d(&[target_h, target_w], false, None, None);
            blended = blended + resized;
        }

        blended / (predictions.len() as f64)
    }
}

/// Evaluation metrics for saliency detection
pub struct SaliencyMetrics;

impl SaliencyMetrics {
    /// Mean Absolute Error
    pub fn mae(pred: &Tensor, target: &Tensor) -> Result<f64, TchError> {
        let pred_sigmoid = pred.sigmoid();
        let mae = (&pred_sigmoid - target).abs().mean(Kind::Float);
        f64::try_from(mae)
    }

    /// F-measure (F1 score)
    pub fn f_measure(pred: &Tensor, target: &Tensor, threshold: f64) ->  Result<f64, TchError> {
        let pred_binary = pred.sigmoid().gt(threshold);
        let target_binary = target.gt(0.5);

        let tp = (&pred_binary * &target_binary).sum(Kind::Float);
        let fp = ((&pred_binary * (1.0 - &target_binary)) as Tensor).sum(Kind::Float);
        let fn_val = (((1.0 - &pred_binary) * &target_binary) as Tensor).sum(Kind::Float);

        let precision = &tp / (&tp + &fp + 1e-8);
        let recall = &tp / (&tp + &fn_val + 1e-8);

        let f_measure = 2.0 * &precision * &recall / (&precision + &recall + 1e-8);
        f64::try_from(f_measure)
    }

    /// S-measure (Structure measure)
    pub fn s_measure(pred: &Tensor, target: &Tensor) -> Result<f64, TchError> {
        let pred_sigmoid = pred.sigmoid();
        let alpha = 0.5;

        // Object-aware structural similarity
        let mean_pred = pred_sigmoid.mean(Kind::Float);
        let mean_target = target.mean(Kind::Float);

        let obj_sim = if f64::try_from(&mean_target)? == 0.0 {
            1.0 - f64::try_from(&mean_pred)?
        } else {
            let sigma_pred = pred_sigmoid.std(false);
            let sigma_target = target.std(false);
            let sigma_cross = ((&pred_sigmoid - &mean_pred) * (target - &mean_target)).mean(Kind::Float);

            let numerator = 4.0 * f64::try_from(&mean_pred)? * f64::try_from(&mean_target)? * f64::try_from(&sigma_cross)?;
            let denominator = (f64::try_from(&mean_pred)?.powi(2) + f64::try_from(&mean_target)?.powi(2))
                * (f64::try_from(&sigma_pred)?.powi(2) + f64::try_from(&sigma_target)?.powi(2));

            if denominator == 0.0 { 0.0 } else { numerator / denominator }
        };

        // Region-aware structural similarity (simplified)
        let region_sim = obj_sim; // Simplified implementation

        Ok(alpha * obj_sim + (1.0 - alpha) * region_sim)
    }

    /// E-measure (Enhanced-alignment measure)
    pub fn e_measure(pred: &Tensor, target: &Tensor) -> Result<f64, TchError> {
        let pred_sigmoid = pred.sigmoid();

        // Enhanced alignment matrix
        let enhanced_matrix = (&pred_sigmoid - &pred_sigmoid.mean(Kind::Float)).abs()
            + (target - target.mean(Kind::Float)).abs();

        let align_matrix = 2.0 * &pred_sigmoid * target / (&pred_sigmoid + target + 1e-8);
        let enhanced_align = &enhanced_matrix * &align_matrix;

        f64::try_from(enhanced_align.mean(Kind::Float))
    }
}

/// Configuration loader for YAML files
#[derive(Debug, Clone)]
pub struct Config {
    pub model: ModelConfig,
    pub training: TrainingConfig,
    pub data: DataConfig,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub backbone: String,
    pub pretrained: bool,
    pub channels: Vec<i64>,
    pub device: Device,
}

#[derive(Debug, Clone)]
pub struct DataConfig {
    pub train_path: String,
    pub val_path: String,
    pub image_size: i64,
    pub augmentation: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            backbone: "res2net50".to_string(),
            pretrained: true,
            channels: vec![256, 512, 1024, 2048],
            device: Device::cuda_if_available(),
        }
    }
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            train_path: "data/train".to_string(),
            val_path: "data/val".to_string(),
            image_size: 352,
            augmentation: true,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            training: TrainingConfig::default(),
            data: DataConfig::default(),
        }
    }
}