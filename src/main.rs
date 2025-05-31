// main.rs - Example usage of the InSPyReNet Rust implementation

use anyhow::Result;
use tch::{Device, Tensor, Kind, nn};
use ocelot::{Ocelot, Config, SaliencyMetrics, ImageProcessor};

fn main() -> Result<()> {
    // Initialize configuration
    let config = Config::default();

    // Create model
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let model = Ocelot::new(&vs.root(), config)?;

    println!("Model created successfully on device: {:?}", device);

    // Example 1: Inference on a single image
    inference_example(&model)?;

    Ok(())
}

fn inference_example(model: &Ocelot) -> Result<()> {
    println!("Running inference example...");

    // Create dummy input (batch_size=1, channels=3, height=352, width=352)
    let input = Tensor::randn(&[1, 3, 352, 352], (Kind::Float, model.config().device));

    // Normalize input
    let normalized_input = ImageProcessor::normalize(&input);

    // Forward pass
    let output = model.forward_t(&normalized_input, false)?;

    // The output should be a saliency map
    println!("Input shape: {:?}", input.size());
    println!("Output shape: {:?}", output.size());

    // Apply sigmoid to get probability map
    let saliency_map = output.sigmoid();
    println!("Saliency map range: [{:.4}, {:.4}]",
             f64::try_from(saliency_map.min())?,
             f64::try_from(saliency_map.max())?);

    Ok(())
}

fn pyramid_processing_example() -> Result<()> {
    println!("Running pyramid processing example...");

    let device = Device::cuda_if_available();
    let image = Tensor::randn(&[1, 3, 512, 512], (Kind::Float, device));

    // Create image pyramid
    let scales = vec![0.5, 0.75, 1.0, 1.25];
    let pyramid = ImageProcessor::create_pyramid(&image, &scales);

    println!("Original image: {:?}", image.size());
    for (i, scale_img) in pyramid.iter().enumerate() {
        println!("Scale {}: {:?}", scales[i], scale_img.size());
    }

    // Simulate predictions for each scale
    let predictions: Vec<Tensor> = pyramid.iter()
        .map(|img| {
            // Dummy prediction (normally would come from model)
            let (_, _, h, w) = img.size4().unwrap();
            Tensor::rand(&[1, 1, h, w], (Kind::Float, device))
        })
        .collect();

    // Blend predictions
    let blended = ImageProcessor::blend_pyramid(&predictions, (512, 512));
    println!("Blended prediction: {:?}", blended.size());

    Ok(())
}

fn evaluation_example() -> Result<()> {
    println!("Running evaluation example...");

    let device = Device::cuda_if_available();

    // Create dummy prediction and ground truth
    let prediction = Tensor::randn(&[1, 1, 352, 352], (Kind::Float, device));
    let ground_truth = Tensor::rand(&[1, 1, 352, 352], (Kind::Float, device)).gt(0.5).to_kind(Kind::Float);

    // Calculate metrics
    let mae = SaliencyMetrics::mae(&prediction, &ground_truth).unwrap_or(0f64);
    let f_measure = SaliencyMetrics::f_measure(&prediction, &ground_truth, 0.5).unwrap_or(0f64);
    let s_measure = SaliencyMetrics::s_measure(&prediction, &ground_truth).unwrap_or(0f64);
    let e_measure = SaliencyMetrics::e_measure(&prediction, &ground_truth).unwrap_or(0f64);

    println!("Evaluation Metrics:");
    println!("MAE: {:.4}", mae);
    println!("F-measure: {:.4}", f_measure);
    println!("S-measure: {:.4}", s_measure);
    println!("E-measure: {:.4}", e_measure);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() -> Result<()> {
        let device = Device::cuda_if_available();
        let config = Config::default();
        let vs = nn::VarStore::new(device);
        let model = Ocelot::new(&vs.root(), config)?;

        // Test forward pass
        let input = Tensor::randn(&[1, 3, 352, 352], (Kind::Float, Device::Cpu));
        let output = model.forward_t(&input, true)?;

        assert_eq!(output.size(), vec![1, 1, 352, 352]);
        Ok(())
    }

    #[test]
    fn test_image_processing() -> Result<()> {
        let image = Tensor::randn(&[1, 3, 224, 224], (Kind::Float, Device::Cpu));

        let normalized = ImageProcessor::normalize(&image);
        let denormalized = ImageProcessor::denormalize(&normalized);

        // Should be approximately equal (within floating point precision)
        let diff = (&image - &denormalized).abs().max();
        assert!(f64::from(diff) < 1e-6);

        Ok(())
    }

    #[test]
    fn test_pyramid_creation() -> Result<()> {
        let image = Tensor::randn(&[1, 3, 256, 256], (Kind::Float, Device::Cpu));
        let scales = vec![0.5, 1.0, 2.0];
        let pyramid = ImageProcessor::create_pyramid(&image, &scales);

        assert_eq!(pyramid.len(), 3);
        assert_eq!(pyramid[0].size(), vec![1, 3, 128, 128]); // 0.5x scale
        assert_eq!(pyramid[1].size(), vec![1, 3, 256, 256]); // 1.0x scale
        assert_eq!(pyramid[2].size(), vec![1, 3, 512, 512]); // 2.0x scale

        Ok(())
    }
}