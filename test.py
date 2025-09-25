import argparse
from groundingdino.util.inference import load_model, load_image, predict, Model
import cv2
import torch
import torchvision.ops as ops
from torchvision.ops import box_convert
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from pathlib import Path



def process_single_image(
        model,
        image_path,
        text_prompt,
        box_threshold,
        text_threshold
):
    image_bytes = open(image_path, "rb").read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_source = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    detections = model.predict_with_caption(
        image=image_source,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )[0]

    if not detections or detections.xyxy.shape[0] == 0:
        print(f"No detections found for {os.path.basename(image_path)}")
        return None, image_source

    boxes = []
    if detections and detections.xyxy.shape[0] > 0:
        for i in range(detections.xyxy.shape[0]):
            box = detections.xyxy[i]
            confidence = detections.confidence[i] if hasattr(detections, "confidence") else 0.0
            boxes.append(
                {
                    "x_min": float(box[0]),
                    "y_min": float(box[1]),
                    "x_max": float(box[2]),
                    "y_max": float(box[3]),
                    "confidence": float(confidence),
                }
            )
        # Sort boxes by confidence in descending order (highest first)
        boxes.sort(key=lambda x: x["confidence"], reverse=True)

    return boxes, image_source


def process_folder(
        model_config,
        model_weights,
        folder_path,
        text_prompt,
        box_threshold,
        text_threshold
):
    # Initialize model once for all images
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = Model(
            model_config_path=model_config,
            model_checkpoint_path=model_weights,
            device=device,
        )

    # Get all image files in the folder
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(folder_path).glob(f"*{ext}"))
        image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"No image files found in {folder_path}")
        return

    print(f"Found {len(image_files)} images to process")
    print(f"Text prompt: {text_prompt}")
    print("-" * 50)

    # Process all images first
    processed_images = []
    for idx, image_path in enumerate(image_files):
        try:
            print(f"Processing [{idx+1}/{len(image_files)}]: {os.path.basename(image_path)}")
            boxes, image_source = process_single_image(
                model=model,
                image_path=str(image_path),
                text_prompt=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )

            # Convert BGR to RGB for matplotlib
            image_rgb = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)

            processed_images.append({
                'path': image_path,
                'image': image_rgb,
                'boxes': boxes
            })

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    if not processed_images:
        print("No images were successfully processed")
        return

    print("-" * 50)
    print(f"Successfully processed {len(processed_images)} images")
    print("Use arrow keys to navigate: ← Previous | → Next | Q to quit")

    # Create interactive display
    class ImageViewer:
        def __init__(self, images, text_prompt):
            self.images = images
            self.text_prompt = text_prompt
            self.current_idx = 0

            # Define color palette for boxes
            self.colors = ['lime', 'cyan', 'magenta', 'yellow', 'orange',
                          'red', 'blue', 'green', 'purple', 'pink']

            self.fig, self.ax = plt.subplots(1, figsize=(12, 8))
            self.fig.canvas.mpl_connect('key_press_event', self.on_key)

            self.update_display()

        def update_display(self):
            self.ax.clear()

            current_data = self.images[self.current_idx]
            self.ax.imshow(current_data['image'])

            # Draw boxes if they exist
            if current_data['boxes']:
                for i, box in enumerate(current_data['boxes'][:5]):  # Show top 10 boxes
                    x_min = box["x_min"]
                    y_min = box["y_min"]
                    width = box["x_max"] - box["x_min"]
                    height = box["y_max"] - box["y_min"]
                    confidence = box["confidence"]

                    # Get color for this box
                    color = self.colors[i % len(self.colors)]

                    # Create a rectangle patch
                    rect = patches.Rectangle((x_min, y_min), width, height,
                                            linewidth=2, edgecolor=color, facecolor='none')
                    self.ax.add_patch(rect)

                    # Add confidence score as text
                    self.ax.text(x_min, y_min - 5, f'{confidence:.2f}',
                                color=color, fontsize=10, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

            self.ax.set_axis_off()
            title = f'[{self.current_idx+1}/{len(self.images)}] {os.path.basename(current_data["path"])} - Detected: {self.text_prompt}'
            self.ax.set_title(title, fontsize=12)

            self.fig.canvas.draw()

        def on_key(self, event):
            if event.key == 'right' and self.current_idx < len(self.images) - 1:
                self.current_idx += 1
                self.update_display()
            elif event.key == 'left' and self.current_idx > 0:
                self.current_idx -= 1
                self.update_display()
            elif event.key == 'q':
                plt.close(self.fig)

    # Create and show the viewer
    viewer = ImageViewer(processed_images, text_prompt)
    plt.show()

    print("Viewer closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test for Grounding DINO - Process folder of images')
    parser.add_argument('--weights', default="weights/groundingdino_swint_ogc.pth", type=str, help='Path to the model weights')
    parser.add_argument('--config', default="groundingdino/config/GroundingDINO_SwinT_OGC.py", type=str, help='Path to the model config')
    parser.add_argument('--folder_path', default="training-data/test", type=str, help='Path to the folder containing images')
    parser.add_argument('--text_prompt', default="PDP.", type=str, help='Text prompt')
    parser.add_argument('--box_threshold', default=0.2, type=float, help='Box threshold')
    parser.add_argument('--text_threshold', default=0.25, type=float, help='Text threshold')
    args = parser.parse_args()

    process_folder(
        model_weights=args.weights,
        model_config=args.config,
        folder_path=args.folder_path,
        text_prompt=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
    )
