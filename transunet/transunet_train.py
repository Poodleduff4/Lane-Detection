import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from timm.models.vision_transformer import vit_base_patch16_224
import torch.nn.functional as F

# Set the environment variable for CUDA
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class CULANEDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
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
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Input size for Vision Transformer
    transforms.ToTensor(),
])

train_dataset = CULANEDataset(
    images_dir='/home/ubuntu/deeplanes/root/images/train/train/train',
    masks_dir='/home/ubuntu/deeplanes/root/ll_seg_annotations/train/train/train',
    transform=transform
)

val_dataset = CULANEDataset(
    images_dir='/home/ubuntu/deeplanes/root/images/val/val/val',
    masks_dir='/home/ubuntu/deeplanes/root/ll_seg_annotations/val/val/val',
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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

        # Resize the output to match the target size
        output = torch.nn.functional.interpolate(output, size=(224, 224), mode="bilinear", align_corners=False)

        return torch.sigmoid(output)

# Instantiate the model
model = TransUNet(img_size=224, in_channels=3, out_channels=1).cuda()

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, model_path='transunet_lane_detection.pth'):
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

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save the model after every epoch
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch 
        }, f'transunet_lane_detection_epoch_{epoch+1}.pth')

        # save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, 'best_transunet_lane_detection.pth')

def tversky_loss(preds, targets, alpha=0.5, beta=0.5, smooth=1e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    true_pos = (preds * targets).sum()
    false_neg = ((1 - preds) * targets).sum()
    false_pos = (preds * (1 - targets)).sum()
    
    tversky_index = (true_pos + smooth) / (true_pos + alpha * false_pos + beta * false_neg + smooth)
    return 1 - tversky_index

# Define Focal Loss
def focal_loss(preds, targets, alpha=0.8, gamma=2):
    bce_loss = F.binary_cross_entropy(preds, targets, reduction='none')
    p_t = torch.exp(-bce_loss)
    focal_loss = alpha * ((1 - p_t) ** gamma) * bce_loss
    return focal_loss.mean()

# Mixed Loss
def mixed_loss(preds, targets, tversky_alpha=0.5, tversky_beta=0.5, focal_alpha=0.8, focal_gamma=2, weight_tversky=0.7):
    tversky = tversky_loss(preds, targets, alpha=tversky_alpha, beta=tversky_beta)
    focal = focal_loss(preds, targets, alpha=focal_alpha, gamma=focal_gamma)
    return weight_tversky * tversky + (1 - weight_tversky) * focal

# Loss and optimizer
criterion = lambda preds, targets: mixed_loss(preds, targets, tversky_alpha=0.3, tversky_beta=0.7, focal_alpha=0.8, focal_gamma=1, weight_tversky=0.7)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)