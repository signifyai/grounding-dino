# Grounding DINO Fine-tuning 🦖


We have expanded on the original DINO  repository 
https://github.com/IDEA-Research/GroundingDINO 
by introducing the capability to train the model with image-to-text grounding. This capability is essential in applications where textual descriptions must align with regions of an image. For instance, when the model is given a caption "a cat on the sofa," it should be able to localize both the "cat" and the "sofa" in the image.

## Features:

- **Fine-tuning DINO**: This extension works allows you to fine-tune DINO on your custom dataset.
- **Bounding Box Regression**: Uses Generalized IoU and Smooth L1 loss for improved bounding box prediction.
- **Position-aware Logit Losses**: The model not only learns to detect objects but also their positions in the captions.
- **NMS**: We also implemented phrase based NMS to remove redundant boxes of same objects


## Installation:
  ```
  git clone https://github.com/signifyai/grounding-dino.git
  cd grounding-dino
  python3.11 -m venv myenv
  source myenv/bin/activate

  pip install -r requirements.txt
  pip install -e . --verbose
  ```

## Train:

1. Prepare your dataset with images and associated textual captions. A tiny dataset is given multimodal-data to demonstrate the expected data format.
2. Run the train.py for training:

```bash
python train.py [OPTIONS]
```

### Train Command-line Arguments:
- `--epochs`: Number of training epochs (default: 500)
- `--save_path`: Directory to save model checkpoints (default: 'weights/trained/')
- `--save_epoch`: Save checkpoint every N epochs (default: 50)
- `--ann_file`: Path to annotation CSV file (default: 'prepared-training-data/annotation/annotation.csv')
- `--images_dir`: Path to training images directory (default: 'prepared-training-data/images')

Example:
```bash
python train.py --epochs 100 --save_epoch 10 --save_path weights/custom/
```

## Test:

Test the model on a folder of images with interactive visualization:

```bash
python test.py --folder_path PATH_TO_IMAGES [OPTIONS]
```

### Test Command-line Arguments:
- `--folder_path`: Path to folder containing images to process (required)
- `--weights`: Path to model weights file (default: 'weights/groundingdino_swint_ogc.pth')
- `--config`: Path to model configuration file (default: 'groundingdino/config/GroundingDINO_SwinT_OGC.py')
- `--text_prompt`: Text prompt for object detection (default: 'PDP.')
- `--box_threshold`: Minimum confidence for bounding boxes (default: 0.2)
- `--text_threshold`: Minimum confidence for text matching (default: 0.25)

Example:
```bash
python test.py --folder_path training-data/test --text_prompt "ingredients section." --box_threshold 0.3
```

### Interactive Navigation:
Once the test script processes all images, use these keyboard controls:
- **→** (Right arrow): Next image
- **←** (Left arrow): Previous image
- **Q**: Quit viewer

The viewer displays bounding boxes in different colors for easy distinction, with confidence scores shown for each detection.