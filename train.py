import argparse
from groundingdino.util.inference import load_model, load_image,train_image
import cv2
import os
import csv
import torch
from collections import defaultdict
import torch.optim as optim
import subprocess
from datetime import datetime

# Check if weights file exists, download if not
weights_dir = "weights"
weights_file = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")

if not os.path.exists(weights_file):
    print(f"[!] Weights file not found at {weights_file}")

    # Create weights directory if it doesn't exist
    os.makedirs(weights_dir, exist_ok=True)

    # Download the weights file using curl
    url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    print(f"[*] Downloading weights from {url}")

    try:
        subprocess.run(["curl", "-L", "-o", weights_file, url], check=True)
        print(f"[+] Successfully downloaded weights to {weights_file}")
    except subprocess.CalledProcessError as e:
        print(f"[!] Error downloading weights: {e}")
        raise
else:
    print(f"[+] Weights file found at {weights_file}")

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Model
model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", weights_file, device=device)

def draw_box_with_label(image, output_path, coordinates, label, color=(0, 0, 255), thickness=2, font_scale=0.5):
    """
    Draw a box and a label on an image using OpenCV.

    Parameters:
    - image (numpyarray): input image.
    - output_path (str): Path to save the image with the box and label.
    - coordinates (tuple): A tuple (x1, y1, x2, y2) indicating the top-left and bottom-right corners of the box.
    - label (str): The label text to be drawn next to the box.
    - color (tuple, optional): Color of the box and label in BGR format. Default is red (0, 0, 255).
    - thickness (int, optional): Thickness of the box's border. Default is 2 pixels.
    - font_scale (float, optional): Font scale for the label. Default is 0.5.
    """
    
    # Draw the rectangle
    cv2.rectangle(image, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), color, thickness)
    
    # Define a position for the label (just above the top-left corner of the rectangle)
    label_position = (coordinates[0], coordinates[1]-10)
    
    # Draw the label
    cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    
    # Save the modified image
    cv2.imwrite(output_path, image)

def read_dataset(ann_file, images_dir):
    ann_Dict= defaultdict(lambda: defaultdict(list))
    with open(ann_file) as file_obj:
        ann_reader= csv.DictReader(file_obj)  
        # Iterate over each row in the csv file
        # using reader object
        for row in ann_reader:
            #print(row)
            img_n=os.path.join(images_dir,row['image_name'])
            x1=int(row['bbox_x1'])
            y1=int(row['bbox_y1'])
            x2=int(row['bbox_x2'])
            y2=int(row['bbox_y2'])
            label=row['label_name']
            ann_Dict[img_n]['boxes'].append([x1,y1,x2,y2])
            ann_Dict[img_n]['captions'].append(label)
    return ann_Dict


def train(model, ann_file, epochs=1, save_path='weights/trained/',save_epoch=50, images_dir='prepared-training-data/images'):
    # Read Dataset
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{save_path}model_{current_time}_epoch_"
    ann_Dict = read_dataset(ann_file, images_dir)
    
    # Add optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    # Ensure the model is in training mode
    model.train()

    for epoch in range(epochs):
        total_loss = 0  # Track the total loss for this epoch
        for idx, (IMAGE_PATH, vals) in enumerate(ann_Dict.items()):
            image_source, image = load_image(IMAGE_PATH)
            bxs = vals['boxes']
            captions = vals['captions']

            # Zero the gradients
            optimizer.zero_grad()
            
            # Call the training function for each image and its annotations
            loss = train_image(
                model=model,
                image_source=image_source,
                image=image,
                caption_objects=captions,
                box_target=bxs,
                device=device
            )
            
            # Backpropagate and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()  # Accumulate the loss
            print(f"Processed image {idx+1}/{len(ann_Dict)}, Loss: {loss.item()}")

        # Print the average loss for the epoch
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss / len(ann_Dict)}")
        if (epoch%save_epoch)==0:
            # Save the model's weights after each epoch
            torch.save(model.state_dict(), f"{save_path}{epoch}.pth")
            print(f"Model weights saved to {save_path}{epoch}.pth")



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Training for Grounding DINO')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--save_path', type=str, default='weights/trained/', help='Path to save the model weights')
    parser.add_argument('--save_epoch', type=int, default=50, help='Number of epochs to save the model weights')
    parser.add_argument('--ann_file', type=str, default='prepared-training-data/annotation/annotation.csv', help='Path to the annotation file')
    parser.add_argument('--images_dir', type=str, default='prepared-training-data/images', help='Path to the images directory')
    args = parser.parse_args()

    train(model=model, ann_file=args.ann_file, epochs=args.epochs, save_path=args.save_path, save_epoch=args.save_epoch, images_dir=args.images_dir)
