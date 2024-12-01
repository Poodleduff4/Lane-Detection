import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from timm.models.vision_transformer import vit_base_patch16_224
import torch.nn.functional as F

# Custom transform for joint image and mask augmentation
class SegmentationTransform:
    def __init__(self, img_size=224):
        self.img_size = img_size

    def __call__(self, image, mask):
        ## Apply consistent random transformations to both image and mask
        if torch.rand(1) > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        ## Random rotation
        angle = transforms.RandomRotation.get_params([-10, 10])
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

        ## Resize
        image = TF.resize(image, (self.img_size, self.img_size))
        mask = TF.resize(mask, (self.img_size, self.img_size))

        ## Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask

class CULANEDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        ## Initialize dataset with paths and transformations
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            ## Resize and convert to tensor
            image = TF.resize(image, (224, 224))
            mask = TF.resize(mask, (224, 224))
            image = TF.to_tensor(image)
            mask = TF.to_tensor(mask)

        return image, mask

# Validation transform
class ValidationTransform:
    def __init__(self, img_size=224):
        self.img_size = img_size

    def __call__(self, image, mask):
        ## Resize and convert to tensor
        image = TF.resize(image, (self.img_size, self.img_size))
        mask = TF.resize(mask, (self.img_size, self.img_size))
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask

# Data transformations
train_transform = SegmentationTransform(img_size=224)
val_transform = ValidationTransform(img_size=224)

train_dataset = CULANEDataset(
    images_dir='/home/ubuntu/deeplanes/root/images/train/train/train',
    masks_dir='/home/ubuntu/deeplanes/root/ll_seg_annotations/train/train/train',
    transform=train_transform
)

val_dataset = CULANEDataset(
    images_dir='/home/ubuntu/deeplanes/root/images/val/val/val',
    masks_dir='/home/ubuntu/deeplanes/root/ll_seg_annotations/val/val/val',
    transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)

# Model definition with adjusted skip connections
class TransUNet(nn.Module):
    def __init__(self, img_size=224, in_channels=3, out_channels=1):
        super(TransUNet, self).__init__()
        self.encoder = vit_base_patch16_224(pretrained=True)
        self.encoder.head = nn.Identity()  # Remove classification head

        ## Indices of transformer blocks to extract features for skip connections
        self.skip_connections = [2, 5, 8, 11]

        ## Convolutional layer to project transformer features to decoder
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

# Loss functions
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

# Instantiate the model
model = TransUNet(img_size=224, in_channels=3, out_channels=1).cuda()

# Define the optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.999))

# Add learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,  # reduce LR by half when plateauing
    patience=3,   # number of epochs to wait before reducing LR
    verbose=True,
    min_lr=1e-6  # minimum LR we'll allow
)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=16):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')
        for images, masks in train_loader_tqdm:
            images = images.cuda()
            masks = masks.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_loader_tqdm.set_postfix({'Loss': train_loss / len(train_loader.dataset)})

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        val_loader_tqdm = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation')
        with torch.no_grad():
            for images, masks in val_loader_tqdm:
                images = images.cuda()
                masks = masks.cuda()

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item() * images.size(0)
                val_loader_tqdm.set_postfix({'Loss': val_loss / len(val_loader.dataset)})

        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # Save the model after every epoch
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch 
        }, f'unet_lane_detection_epoch_{epoch+1}.pth')

        # Additionally, save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, 'best_unet_lane_detection.pth')

        # Update learning rate
        scheduler.step(val_loss)

# Define the criterion using the combined loss
criterion = combined_loss

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10)
