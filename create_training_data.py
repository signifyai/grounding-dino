import os
import argparse
import shutil
import xml.etree.ElementTree as ET
import csv

# === CONFIG ===
INPUT_DIR           = 'training-data'  # Contains train/, test/, valid/ subdirs
OUTPUT_DIR          = 'prepared-training-data'
CSV_OUTPUT_FILENAME = 'annotation.csv'  # will be placed under annotation/

# === UTILITIES ===

def xmls_to_csv_from_paths(xml_paths: list, output_csv: str):
    """Parse all PascalVOC‐style XMLs from given paths → single CSV."""
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'label_name',
            'bbox_x1','bbox_y1','bbox_x2','bbox_y2',
            'image_name',
            'image_width','image_height'
        ])

        xml_count = 0
        for xml_path in xml_paths:
            xml_count += 1
            tree = ET.parse(xml_path)
            root = tree.getroot()

            image_name = root.findtext('filename')
            size = root.find('size')
            width  = size.findtext('width')
            height = size.findtext('height')

            for obj in root.findall('object'):
                label = obj.findtext('name')
                b = obj.find('bndbox')
                xmin = b.findtext('xmin')
                ymin = b.findtext('ymin')
                xmax = b.findtext('xmax')
                ymax = b.findtext('ymax')

                writer.writerow([
                    label,
                    xmin, ymin, xmax, ymax,
                    image_name,
                    width, height
                ])
        print(f"[+] Processed {xml_count} XML files")
    print(f"[+] Wrote CSV annotations to {output_csv}")


def xmls_to_csv(annotations_dir: str, output_csv: str):
    """Parse all PascalVOC‐style XMLs in `annotations_dir` → single CSV."""
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'label_name',
            'bbox_x1','bbox_y1','bbox_x2','bbox_y2',
            'image_name',
            'image_width','image_height'
        ])

        xml_count = 0
        for fname in os.listdir(annotations_dir):
            if not fname.lower().endswith('.xml'):
                continue
            xml_path = os.path.join(annotations_dir, fname)
            xml_count += 1
            tree = ET.parse(xml_path)
            root = tree.getroot()

            image_name = root.findtext('filename')
            size = root.find('size')
            width  = size.findtext('width')
            height = size.findtext('height')

            for obj in root.findall('object'):
                label = obj.findtext('name')
                b = obj.find('bndbox')
                xmin = b.findtext('xmin')
                ymin = b.findtext('ymin')
                xmax = b.findtext('xmax')
                ymax = b.findtext('ymax')

                writer.writerow([
                    label,
                    xmin, ymin, xmax, ymax,
                    image_name,
                    width, height
                ])
        print(f"[+] Processed {xml_count} XML files")
    print(f"[+] Wrote CSV annotations to {output_csv}")


def mkdirs_safe(path):
    os.makedirs(path, exist_ok=True)


# === MAIN ===

def main(input_dir=INPUT_DIR):

    # clear directories
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    # 1) Create output directory structure
    mkdirs_safe(os.path.join(OUTPUT_DIR, 'images'))
    mkdirs_safe(os.path.join(OUTPUT_DIR, 'annotation'))
    mkdirs_safe(os.path.join(OUTPUT_DIR, 'test_images'))

    # 2) Process train and valid directories - copy images and collect XMLs
    total_train_images = 0

    for split in ['train', 'valid']:
        split_dir = os.path.join(input_dir, split)
        split_images = [f for f in os.listdir(split_dir)
                       if f.lower().endswith(('.png','.jpg','.jpeg'))]

        print(f"[*] Found {len(split_images)} {split} images")

        for img_name in split_images:
            # Copy image to images/
            src_img = os.path.join(split_dir, img_name)
            dst_img = os.path.join(OUTPUT_DIR, 'images', img_name)
            shutil.copy2(src_img, dst_img)

        total_train_images += len(split_images)
        print(f"[+] Copied {len(split_images)} {split} images → {OUTPUT_DIR}/images")

    print(f"[+] Total training images (train + valid): {total_train_images}")

    # 3) Process test directory - copy images to test_images/
    test_dir = os.path.join(input_dir, 'test')
    test_images = [f for f in os.listdir(test_dir)
                  if f.lower().endswith(('.png','.jpg','.jpeg'))]

    print(f"[*] Found {len(test_images)} test images")

    for img_name in test_images:
        src_img = os.path.join(test_dir, img_name)
        dst_img = os.path.join(OUTPUT_DIR, 'test_images', img_name)
        shutil.copy2(src_img, dst_img)

    print(f"[+] Copied {len(test_images)} test images → {OUTPUT_DIR}/test_images")

    # 4) Generate CSV from all XMLs in training-data/train and valid
    csv_outpath = os.path.join(OUTPUT_DIR, 'annotation', CSV_OUTPUT_FILENAME)

    # Combine XMLs from both train and valid directories
    all_xml_data = []
    for split in ['train', 'valid']:
        split_dir = os.path.join(input_dir, split)
        for fname in os.listdir(split_dir):
            if fname.lower().endswith('.xml'):
                xml_path = os.path.join(split_dir, fname)
                all_xml_data.append(xml_path)

    print(f"[*] Found {len(all_xml_data)} total XML files to process")
    xmls_to_csv_from_paths(all_xml_data, csv_outpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create training data')
    parser.add_argument('--input_dir', type=str, default=INPUT_DIR, help='Path to the input directory')
    args = parser.parse_args()
    main(input_dir=args.input_dir)