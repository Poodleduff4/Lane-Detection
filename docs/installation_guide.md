# Installation Guide

Follow these steps if you prefer a step-by-step walk-through instead of the short Quick Start in the README.

1. Clone the repository
   ```bash
   git clone https://github.com/Poodleduff4/Lane-Detection.git
   cd Lane-Detection
   ```
2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Download the pre-trained weights (links in the report) and place them in the project root.
5. Run an example inference
   ```bash
   python transunet_inference.py --image images/example.jpg
   ```

For advanced usage (training, video overlay, YOLO baseline) see the README. 