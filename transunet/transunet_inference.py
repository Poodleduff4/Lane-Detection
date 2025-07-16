import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from torch.cuda.amp import autocast
from skimage.filters import threshold_otsu
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from timm.models.vision_transformer import vit_base_patch16_224

# Set the environment variable for CUDA
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class CULANEDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_list = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.image_list[idx])
        
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

class TransUNet(nn.Module):
    def __init__(self, img_size=224, in_channels=3, out_channels=1):
        super(TransUNet, self).__init__()
        self.encoder = vit_base_patch16_224(pretrained=True)  # Pretrained Vision Transformer
        self.encoder.head = nn.Identity()  # Remove classification head

        self.conv_head = nn.Conv2d(768, 512, kernel_size=1)  # Reduce transformer feature dims
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

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
        output = torch.nn.functional.interpolate(output, size=(224, 224), mode="bilinear", align_corners=False)

        return torch.sigmoid(output)

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Input size for Vision Transformer
    transforms.ToTensor(),
])

# Load the trained model
model = TransUNet(img_size=224, in_channels=3, out_channels=1).cuda()
model.load_state_dict(torch.load('./good_models/unet_lane_detection_epoch_4.pth')['model_state_dict'])
model.eval()

# Evaluation
val_image_dir = '/home/ubuntu/deeplanes/root/images/val/val/val'
val_mask_dir = '/home/ubuntu/deeplanes/root/ll_seg_annotations/val/val/val'
val_dataset = CULANEDataset(val_image_dir, val_mask_dir, transform)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

criterion = nn.BCEWithLogitsLoss()

@torch.no_grad()
def evaluate():
    model.eval()
    running_loss = 0.0
    for images, masks in tqdm(val_loader, desc='Evaluating'):
        images, masks = images.cuda(), masks.cuda()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        running_loss += loss.item() * images.size(0)
    print(f'Validation Loss: {running_loss / len(val_loader.dataset):.4f}')

evaluate() 