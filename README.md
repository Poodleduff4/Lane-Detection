
# CULANE - Lane Detection Project - TMU

- Transformer-UNET Lane Detection Project

  Major modifications to the code involved:
  - replacing the standard unet model with a transformer-unet due to poor performance on CULANE
  - switching loss functions away from BCE loss to a tversky/focal loss function
  - moving away from CV2 for video inference

## Notes

- If you are running video/YOLO inference, you will need to modify model_1/model_2.py's file name to be model.py in order for the inference files to correctly realize the model architecture currently being run by the video inference code and modify the path within the file to point to the file being run.
