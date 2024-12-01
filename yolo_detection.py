import imageio.v3 as iio
from tqdm import tqdm
import numpy as np
import cv2
import imageio
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import TransUNet
from ultralytics import YOLO

def process_frame(frame, transunet_model, yolo_model, transform):
    ## Convert frame (numpy array) to PIL Image
    pil_image = Image.fromarray(frame)

    ## Apply transformations for TransUNet
    input_tensor = transform(pil_image).unsqueeze(0)

    ## Run inference on TransUNet
    with torch.no_grad():
        output = transunet_model(input_tensor)

    ## Convert TransUNet output to numpy array and normalize
    mask = output.squeeze().cpu().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min())  # Normalize to [0, 1]

    ## Resize mask to match frame size
    mask = np.array(Image.fromarray(mask).resize((frame.shape[1], frame.shape[0])))

    ## Invert the mask
    inverted_mask = mask

    ## Create green overlay
    green_overlay = np.zeros_like(frame, dtype=np.float32)
    green_overlay[:, :, 1] = 255.0  # Set green channel to 255

    ## Apply mask to the green overlay
    green_mask = (np.expand_dims(inverted_mask, axis=-1) * green_overlay)

    ## Blend the original frame with the green mask
    alpha = 0.5  # Adjust transparency
    result_frame = (frame.astype(np.float32) * (1 - alpha)) + (green_mask * alpha)

    ## Clip values to valid range and convert to uint8
    result_frame = np.clip(result_frame, 0, 255).astype(np.uint8)

    ## Run YOLO inference on the frame
    yolo_results = yolo_model.predict(frame)

    ## Draw YOLO detections on the frame
    for result in yolo_results:
        for detection in result.boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box coordinates
            conf = detection.conf[0]  # Confidence score
            cls = int(detection.cls[0])  # Class ID
            label = f"{yolo_model.names[cls]} {conf:.2f}"

            ## Draw bounding box and label
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box
            cv2.putText(result_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return result_frame

def main():
    ## Load the TransUNet model
    transunet_model = TransUNet(in_channels=3, out_channels=1)
    transunet_model.load_state_dict(torch.load('unet_lane_detection_epoch_4.pth', map_location=torch.device('cuda'))['model_state_dict'])
    transunet_model.eval()

    ## Load YOLO model (using YOLOv5 or YOLOv8)
    yolo_model = YOLO('yolov8n.pt')

    ## Define transform for TransUNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    ## Read and process video frame-by-frame
    reader = iio.imiter("snow-vid.mov", plugin="pyav")  # Use PyAV as backend
    metadata = iio.immeta("snow-vid.mov", plugin="pyav")  # Get metadata for the video (e.g., frame count)
    frame_count = metadata.get("nframes", 0)  # Estimate total frames if available

    ## Open output video writer
    fps = metadata.get("fps", 30)
    writer = imageio.get_writer("vid.mp4", fps=fps)

    for frame in tqdm(reader, total=frame_count, desc="Processing video"):
        result_frame = process_frame(frame, transunet_model, yolo_model, transform)
        writer.append_data(result_frame)

    writer.close()

if __name__ == '__main__':
    main()