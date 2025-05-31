use tch::{nn, nn::Module, Tensor, Device};
use anyhow::Result;
use tch::nn::ModuleT;
use crate::{Config, ops::*};

/// Trait for decoder modules
pub trait DecoderModule {
    fn forward_t(&self, features: &[Tensor], train: bool) -> Result<Tensor>;
}

/// Pyramid decoder for InSPyReNet
pub struct PyramidDecoder {
    fpn: FeaturePyramidNetwork,
    pyramid_heads: Vec<PyramidHead>,
    final_conv: nn::Conv2D,
    config: Config,
}

impl PyramidDecoder {
    pub fn new(vs: &nn::Path, config: &Config) -> Result<Self> {
        // Feature channels from backbone
        let in_channels = vec![64, 256, 512, 1024, 2048]; // Typical ResNet/Res2Net channels
        let out_channels = 256;

        let fpn = FeaturePyramidNetwork::new(&(vs / "fpn"), &in_channels, out_channels)?;

        let mut pyramid_heads = Vec::new();
        for i in 0..5 {
            let head = PyramidHead::new(&(vs / format!("head_{}", i)), out_channels)?;
            pyramid_heads.push(head);
        }

        let final_conv = conv2d(&(vs / "final_conv"), out_channels, 1, 1, 1, 0);

        Ok(Self {
            fpn,
            pyramid_heads,
            final_conv,
            config: config.clone(),
        })
    }
}

impl DecoderModule for PyramidDecoder {
    fn forward_t(&self, features: &[Tensor], train: bool) -> Result<Tensor> {
        // Feature pyramid network
        let fpn_features = self.fpn.forward(features)?;

        let mut pyramid_outputs = Vec::new();
        let base_size = (self.config.base_size.0, self.config.base_size.1);

        // Process each pyramid level
        for (i, (feature, head)) in fpn_features.iter().zip(self.pyramid_heads.iter()).enumerate() {
            let mut output = head.forward_t(feature, train)?;

            // Upsample to base size
            if output.size()[2] != base_size.0 || output.size()[3] != base_size.1 {
                output = interpolate_bilinear(&output, base_size);
            }

            pyramid_outputs.push(output);
        }

        // Aggregate pyramid outputs
        let aggregated = self.aggregate_pyramid_outputs(&pyramid_outputs)?;

        // Final prediction
        Ok(self.final_conv.forward(&aggregated))
    }
}

impl PyramidDecoder {
    fn aggregate_pyramid_outputs(&self, outputs: &[Tensor]) -> Result<Tensor> {
        if outputs.is_empty() {
            return Err(anyhow::anyhow!("No pyramid outputs to aggregate"));
        }

        // Weighted sum of pyramid levels
        let weights = vec![0.1, 0.2, 0.4, 0.2, 0.1]; // Higher weight for middle scales
        let mut result = Tensor::zeros_like(&outputs[0]);

        for (output, &weight) in outputs.iter().zip(weights.iter()) {
            result = result + output * weight;
        }

        Ok(result)
    }
}

/// Feature Pyramid Network
pub struct FeaturePyramidNetwork {
    lateral_convs: Vec<nn::Conv2D>,
    fpn_convs: Vec<nn::Conv2D>,
    out_channels: i64,
}

impl FeaturePyramidNetwork {
    pub fn new(vs: &nn::Path, in_channels: &[i64], out_channels: i64) -> Result<Self> {
        let mut lateral_convs = Vec::new();
        let mut fpn_convs = Vec::new();

        for (i, &in_ch) in in_channels.iter().enumerate() {
            // Lateral connections
            let lateral = conv2d(&(vs / format!("lateral_{}", i)), in_ch, out_channels, 1, 1, 0);
            lateral_convs.push(lateral);

            // FPN convolutions
            let fpn_conv = conv2d(&(vs / format!("fpn_{}", i)), out_channels, out_channels, 3, 1, 1);
            fpn_convs.push(fpn_conv);
        }

        Ok(Self {
            lateral_convs,
            fpn_convs,
            out_channels,
        })
    }

    pub fn forward(&self, features: &[Tensor]) -> Result<Vec<Tensor>> {
        if features.len() != self.lateral_convs.len() {
            return Err(anyhow::anyhow!("Feature count mismatch"));
        }

        // Lateral connections
        let mut laterals = Vec::new();
        for (feature, lateral_conv) in features.iter().zip(self.lateral_convs.iter()) {
            laterals.push(lateral_conv.forward(feature));
        }

        // Top-down pathway
        let mut fpn_features = Vec::with_capacity(laterals.len());
        for _ in 0..laterals.len() {
            fpn_features.push(Tensor::zeros_like(&laterals[0]));
        }
        fpn_features[laterals.len() - 1] = laterals[laterals.len() - 1].shallow_clone();

        for i in (0..laterals.len() - 1).rev() {
            let upsampled = interpolate_bilinear(
                &fpn_features[i + 1],
                (laterals[i].size()[2], laterals[i].size()[3])
            );
            fpn_features[i] = laterals[i].shallow_clone() + upsampled;
        }

        // Apply FPN convolutions
        let mut outputs = Vec::new();
        for (feature, fpn_conv) in fpn_features.iter().zip(self.fpn_convs.iter()) {
            outputs.push(fpn_conv.forward(feature));
        }

        Ok(outputs)
    }
}

/// Pyramid head for each scale
pub struct PyramidHead {
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    conv2: nn::Conv2D,
    bn2: nn::BatchNorm,
    conv3: nn::Conv2D,
    attention: ChannelAttention,
}

impl PyramidHead {
    pub fn new(vs: &nn::Path, in_channels: i64) -> Result<Self> {
        let conv1 = conv2d(&(vs / "conv1"), in_channels, in_channels, 3, 1, 1);
        let bn1 = batch_norm2d(&(vs / "bn1"), in_channels);

        let conv2 = conv2d(&(vs / "conv2"), in_channels, in_channels, 3, 1, 1);
        let bn2 = batch_norm2d(&(vs / "bn2"), in_channels);

        let conv3 = conv2d(&(vs / "conv3"), in_channels, in_channels, 1, 1, 0);

        let attention = ChannelAttention::new(&(vs / "attention"), in_channels)?;

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            attention,
        })
    }

    pub fn forward_t(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let mut out = self.conv1.forward(x);
        out = self.bn1.forward_t(&out, train);
        out = out.relu();

        out = self.conv2.forward(&out);
        out = self.bn2.forward_t(&out, train);
        out = out.relu();

        // Channel attention
        let attention_weights = self.attention.forward(&out)?;
        out = out * attention_weights;

        out = self.conv3.forward(&out);

        Ok(out)
    }
}

/// Channel attention module
pub struct ChannelAttention {
    avg_pool: (Tensor, Tensor),
    max_pool: (Tensor, Tensor),
    fc1: nn::Linear,
    fc2: nn::Linear,
    channels: i64,
}

impl ChannelAttention {
    pub fn new(vs: &nn::Path, channels: i64) -> Result<Self> {
        let reduction = 16;
        let fc1 = nn::linear(&(vs / "fc1"), channels, channels / reduction, Default::default());
        let fc2 = nn::linear(&(vs / "fc2"), channels / reduction, channels, Default::default());

        Ok(Self {
            avg_pool: (Tensor::from(1), Tensor::from(1)),
            max_pool: (Tensor::from(1), Tensor::from(1)),
            fc1,
            fc2,
            channels,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let b = x.size()[0];

        // Average pooling branch
        let avg_out = adaptive_avg_pool2d(x, (1, 1));
        let avg_out = avg_out.view([b, self.channels]);
        let avg_out = self.fc1.forward(&avg_out).relu();
        let avg_out = self.fc2.forward(&avg_out);

        // Max pooling branch
        let max_out = x.adaptive_max_pool2d(&[1, 1]);
        let max_out = max_out.0.view([b, self.channels]);
        let max_out = self.fc1.forward(&max_out).relu();
        let max_out = self.fc2.forward(&max_out);

        // Combine and apply sigmoid
        let attention = (avg_out + max_out).sigmoid();
        let attention = attention.view([b, self.channels, 1, 1]);

        Ok(attention)
    }
}

/// Simple decoder alternative
pub struct SimpleDecoder {
    conv_layers: Vec<DecoderBlock>,
    final_conv: nn::Conv2D,
}

impl SimpleDecoder {
    pub fn new(vs: &nn::Path, in_channels: &[i64]) -> Result<Self> {
        let mut conv_layers = Vec::new();

        // Decoder blocks with skip connections
        let decoder_channels = vec![512, 256, 128, 64, 32];

        for (i, (&in_ch, &out_ch)) in in_channels.iter().rev()
            .zip(decoder_channels.iter()).enumerate() {
            let block = DecoderBlock::new(
                &(vs / format!("decoder_{}", i)),
                in_ch,
                out_ch,
            )?;
            conv_layers.push(block);
        }

        let final_conv = conv2d(&(vs / "final"), 32, 1, 1, 1, 0);

        Ok(Self {
            conv_layers,
            final_conv,
        })
    }
}

impl DecoderModule for SimpleDecoder {

    fn forward_t(&self, features: &[Tensor], train: bool) -> Result<Tensor> {
        let mut x = features[features.len() - 1].shallow_clone();

        for (i, block) in self.conv_layers.iter().enumerate() {
            x = block.forward_t(&x, train)?;

            // Skip connection if available
            if i < features.len() - 1 {
                let skip_idx = features.len() - 2 - i;
                let skip = &features[skip_idx];

                // Resize x to match skip connection
                if x.size()[2] != skip.size()[2] || x.size()[3] != skip.size()[3] {
                    x = interpolate_bilinear(&x, (skip.size()[2], skip.size()[3]));
                }

                x = x + skip;
            }
        }

        Ok(self.final_conv.forward(&x))
    }
}

/// Decoder block with upsampling
pub struct DecoderBlock {
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    conv2: nn::Conv2D,
    bn2: nn::BatchNorm,
    upsample: nn::ConvTranspose2D,
}

impl DecoderBlock {
    pub fn new(vs: &nn::Path, in_channels: i64, out_channels: i64) -> Result<Self> {
        let conv1 = conv2d(&(vs / "conv1"), in_channels, out_channels, 3, 1, 1);
        let bn1 = batch_norm2d(&(vs / "bn1"), out_channels);

        let conv2 = conv2d(&(vs / "conv2"), out_channels, out_channels, 3, 1, 1);
        let bn2 = batch_norm2d(&(vs / "bn2"), out_channels);

        let upsample = nn::conv_transpose2d(
            &(vs / "upsample"),
            out_channels,
            out_channels,
            2,
            nn::ConvTransposeConfigND {
                stride: 2,
                padding: 0,
                ..Default::default()
            }
        );

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            upsample,
        })
    }

    pub fn forward_t(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let mut out = self.conv1.forward(x);
        out = self.bn1.forward_t(&out, train);
        out = out.relu();

        out = self.conv2.forward(&out);
        out = self.bn2.forward_t(&out, train);
        out = out.relu();

        out = self.upsample.forward(&out);

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device};

    #[test]
    fn test_fpn_creation() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let in_channels = vec![64, 128, 256, 512];
        let out_channels = 256;

        let fpn = FeaturePyramidNetwork::new(&vs.root(), &in_channels, out_channels);
        assert!(fpn.is_ok());
    }

    #[test]
    fn test_pyramid_head_creation() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let in_channels = 256;

        let head = PyramidHead::new(&vs.root(), in_channels);
        assert!(head.is_ok());
    }

    #[test]
    fn test_channel_attention() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let channels = 256;

        let attention = ChannelAttention::new(&vs.root(), channels);
        assert!(attention.is_ok());
    }
}