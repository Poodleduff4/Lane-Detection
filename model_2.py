# model.py

import torch
import torch.nn as nn
from timm.models.vision_transformer import vit_base_patch16_224

class TransUNet(nn.Module):
    def __init__(self, img_size=224, in_channels=3, out_channels=1):
        super(TransUNet, self).__init__()
        ## Initialize with pretrained Vision Transformer backbone
        self.encoder = vit_base_patch16_224(pretrained=True)
        self.encoder.head = nn.Identity()  # Remove classification head

        ## Reduce transformer feature dimensions
        self.conv_head = nn.Conv2d(768, 512, kernel_size=1)

        ## Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        ## Upsampling layers
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        ## Decoder layers
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        ## Final convolution layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder.patch_embed(x)
        cls_token = self.encoder.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.encoder.pos_embed
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)

        x = x[:, 1:]  # Exclude CLS token
        x = x.permute(0, 2, 1).reshape(batch_size, 768, 14, 14)  # Reshape to spatial
        x = self.conv_head(x)

        bottleneck = self.bottleneck(x)
        dec3 = self.upconv3(bottleneck)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(dec1)

        output = self.final_conv(dec1)

        ## Resize the output to match the target size
        output = torch.nn.functional.interpolate(output, size=(224, 224), mode="bilinear", align_corners=False)

        return torch.sigmoid(output)