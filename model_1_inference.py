import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from timm.models.vision_transformer import vit_base_patch16_224
from torch.cuda.amp import autocast
import torch.nn.functional as F

## Set device for CUDA acceleration
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class TransUNet(nn.Module):
    def __init__(self, img_size=224, in_channels=3, out_channels=1):
        super(TransUNet, self).__init__()
        self.encoder = vit_base_patch16_224(pretrained=True)
        ## Remove classification head from ViT
        self.encoder.head = nn.Identity()

        ## Extract features at different depths for skip connections
        self.skip_connections = [2, 5, 8, 11]

        ## Project transformer features to decoder dimensions
        self.conv_head = nn.Conv2d(768, 512, kernel_size=1)

        # Decoder upsampling layers
        self.up_blocks = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        ])

        # Convolutions to adjust skip connections channels
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(768, 512, kernel_size=1),
            nn.Conv2d(768, 256, kernel_size=1),
            nn.Conv2d(768, 128, kernel_size=1),
            nn.Conv2d(768, 64, kernel_size=1),
        ])

        # Decoder blocks
        self.dec_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256 + 512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(128 + 256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(64 + 128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(32 + 64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(inplace=True),
            ),
        ])

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

        # Initialize final_conv bias to zero
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder.patch_embed(x)  # [B, num_patches, embed_dim]
        cls_token = self.encoder.cls_token.expand(batch_size, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_token, x), dim=1)  # [B, 1 + num_patches, embed_dim]
        x = x + self.encoder.pos_embed  # [B, 1 + num_patches, embed_dim]

        skip_feats = []
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in self.skip_connections:
                # Exclude cls_token
                feat = x[:, 1:, :]
                # Reshape to spatial feature map
                feat = feat.permute(0, 2, 1).contiguous()
                size = int(feat.size(2) ** 0.5)
                feat = feat.view(batch_size, -1, size, size)
                skip_feats.append(feat)

        x = self.encoder.norm(x)
        # Exclude cls_token
        x = x[:, 1:, :]
        # Reshape to spatial feature map
        x = x.permute(0, 2, 1).contiguous()
        size = int(x.size(2) ** 0.5)
        x = x.view(batch_size, -1, size, size)
        x = self.conv_head(x)  # [B, 512, H, W]

        # Build decoder with skip connections
        for i in range(len(self.up_blocks)):
            x = self.up_blocks[i](x)  # Upsample
            skip = skip_feats[-(i+1)]  # Get corresponding skip feature
            # Adjust skip connection channels
            skip = self.skip_convs[i](skip)
            # Resize skip if necessary
            if x.size() != skip.size():
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.dec_blocks[i](x)

        output = self.final_conv(x)
        # Apply sigmoid to produce probabilities between 0 and 1
        output = torch.sigmoid(output)
        return output

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Input size for Vision Transformer
    transforms.ToTensor(),
])

# Custom dataset class
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

# Load the trained model
model = TransUNet(img_size=224, in_channels=3, out_channels=1).cuda()
model.load_state_dict(torch.load('./unet_lane_detection_epoch_10.pth')['model_state_dict'])
model.eval()

# Define loss function
def tversky_loss(preds, targets, alpha=0.3, beta=0.7, smooth=1e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    true_pos = (preds * targets).sum()
    false_neg = ((1 - preds) * targets).sum()
    false_pos = (preds * (1 - targets)).sum()
    
    tversky_index = (true_pos + smooth) / (true_pos + alpha * false_pos + beta * false_neg + smooth)
    return 1 - tversky_index

def focal_loss(preds, targets, alpha=0.8, gamma=1):
    bce_loss = F.binary_cross_entropy(preds, targets, reduction='none')
    p_t = torch.exp(-bce_loss)
    focal_loss = alpha * ((1 - p_t) ** gamma) * bce_loss
    return focal_loss.mean()

def combined_loss(preds, targets, w_bce=0.2, w_tversky=0.5, w_focal=0.3):
    bce = nn.BCELoss()(preds, targets)
    tversky = tversky_loss(preds, targets, alpha=0.3, beta=0.7)
    focal = focal_loss(preds, targets, alpha=0.8, gamma=1)
    total_loss = w_bce * bce + w_tversky * tversky + w_focal * focal
    return total_loss

# Use the same criterion as in training
criterion = BCELoss()

# Load validation data
val_image_dir = '/home/ubuntu/deeplanes/root/images/val/val/val'
val_mask_dir = '/home/ubuntu/deeplanes/root/ll_seg_annotations/val/val/val'
val_dataset = CULANEDataset(val_image_dir, val_mask_dir, transform)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Evaluation function
def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    val_f1 = 0.0
    val_jaccard = 0.0
    num_batches = len(val_loader)
    
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Evaluating'):
            images = images.cuda()
            masks = masks.cuda()

            with autocast(): 
                outputs = model(images)
                loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            preds = outputs.cpu().numpy() > 0.5
            true = masks.cpu().numpy() > 0.5
            
            val_accuracy += accuracy_score(true.flatten(), preds.flatten())
            val_f1 += f1_score(true.flatten(), preds.flatten())
            val_jaccard += jaccard_score(true.flatten(), preds.flatten())

    # Average metrics over all batches
    val_loss /= num_batches
    val_accuracy /= num_batches
    val_f1 /= num_batches
    val_jaccard /= num_batches
    
    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.4f}')
    print(f'Validation F1 Score: {val_f1:.4f}')
    print(f'Validation Jaccard Score: {val_jaccard:.4f}')

# Run evaluation
evaluate(model, val_loader, criterion)
