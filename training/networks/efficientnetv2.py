# path=DeepfakeBench/training/networks/efficientnetv2.py
import torch
import torch.nn as nn
import timm
from metrics.registry import BACKBONE

@BACKBONE.register_module(module_name="efficientnetv2")
class EfficientNetV2(nn.Module):
    def __init__(self, efficientnetv2_config):
        super(EfficientNetV2, self).__init__()
        """ Constructor
        Args:
            efficientnetv2_config: configuration file with the dict format
        """
        self.num_classes = efficientnetv2_config["num_classes"]
        inc = efficientnetv2_config["inc"]
        self.dropout = efficientnetv2_config["dropout"]
        self.mode = efficientnetv2_config["mode"]
        self.variant = efficientnetv2_config.get("variant", "s")  # s, m, or l

        # Map variant to model name
        variant_map = {
            "s": "tf_efficientnetv2_s.in21k",
            "m": "tf_efficientnetv2_m.in21k",
            "l": "tf_efficientnetv2_l.in21k"
        }
        model_name = variant_map[self.variant]

        # Load the EfficientNetV2 model with pre-trained weights
        self.efficientnet = timm.create_model(model_name, pretrained=True)

        # Modify the first convolutional layer to accept input tensors with 'inc' channels
        if hasattr(self.efficientnet, 'conv_stem'):
            self.efficientnet.conv_stem = nn.Conv2d(inc, self.efficientnet.conv_stem.out_channels, 
                                                   kernel_size=3, stride=2, bias=False)

        # Remove the classifier
        self.efficientnet.reset_classifier(0)

        if self.dropout:
            # Add dropout layer if specified
            self.dropout_layer = nn.Dropout(p=self.dropout)

        # Get the feature dimension based on the variant
        feature_dims = {
            "s": 1280,
            "m": 1280,
            "l": 1280
        }
        self.feature_dim = feature_dims[self.variant]

        # Initialize the last_layer layer
        self.last_layer = nn.Linear(self.feature_dim, self.num_classes)

        if self.mode == 'adjust_channel':
            self.adjust_channel = nn.Sequential(
                nn.Conv2d(self.feature_dim, 512, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )

    def features(self, x):
        # Extract features from the EfficientNetV2 model
        x = self.efficientnet.forward_features(x)
        if self.mode == 'adjust_channel':
            x = self.adjust_channel(x)
        return x

    def classifier(self, x):
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        # Apply dropout if specified
        if self.dropout:
            x = self.dropout_layer(x)

        # Apply last_layer layer
        y = self.last_layer(x)
        return y

    def forward(self, x):
        # Extract features and apply classifier layer
        x = self.features(x)
        x = self.classifier(x)
        return x