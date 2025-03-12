import os
import random
import shutil
from tqdm import tqdm
import xml.etree.ElementTree as ET
import yaml

DATASET_PATH = "dataset_20210629145407_top_600"
ANNOTATIONS_FOLDER_PATH = DATASET_PATH + "/annotations"
IMAGES_FOLDER_PATH = DATASET_PATH + "/images"


# Reduce the dataset to a smaller size with a defined number of images
def reduce_dataset(image_num, new_dataset_path):
    # Create output directories
    os.makedirs(new_dataset_path + "/annotations", exist_ok=True)
    os.makedirs(new_dataset_path + "/images", exist_ok=True)

    # Randomly select image_num images
    all_images = [f for f in os.listdir(IMAGES_FOLDER_PATH) if f.endswith(".jpg")]
    selected_images = random.sample(all_images, image_num)

    # Copy selected images and their annotations to the new dataset directory
    for img in tqdm(selected_images, desc="Copying selected images"):
        shutil.copy(f"{IMAGES_FOLDER_PATH}/{img}", f"{new_dataset_path}/images/{img}")

        # Copy corresponding XML annotation file
        xml_file = img.replace(".jpg", ".xml")
        if os.path.exists(f"{ANNOTATIONS_FOLDER_PATH}/{xml_file}"):
            new_xml_path = f"{new_dataset_path}/annotations"
            convert_label_and_copy(xml_file, "lego", new_xml_path)
        else:
            print(f"Annotation file for {img} not found")


# Convert unique labels to a single label and write the file to a new directory
def convert_label_and_copy(xml_file, new_label, output_path):
    tree = ET.parse(os.path.join(ANNOTATIONS_FOLDER_PATH, xml_file))
    root = tree.getroot()

    # Convert to single label
    for obj in root.findall("object"):
        obj.find("name").text = new_label

    tree.write(os.path.join(output_path, xml_file))


# Convert PASCAL VOC (XML) annotation files to YOLO format txt files
def convert_xml_to_yolo_format(lego_dataset_path):
    xml_path = f"{lego_dataset_path}/annotations"
    yolo_txt_path = f"{lego_dataset_path}/labels"
    os.makedirs(yolo_txt_path, exist_ok=True)

    class_mapping = {"lego": 0}

    # Convert XML to YOLO format txt files
    xml_files = [f for f in os.listdir(xml_path) if f.endswith(".xml")]
    for xml_file in tqdm(xml_files, desc="Converting XML to YOLO format TXT", unit="file"):
        tree = ET.parse(os.path.join(xml_path, xml_file))
        root = tree.getroot()

        # Get image dimensions
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)

        yolo_data = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            class_id = class_mapping[class_name]

            # Get bounding box
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # Convert to YOLO format (normalized)
            x_center = (xmin + xmax) / (2 * width)
            y_center = (ymin + ymax) / (2 * height)
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            yolo_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

        # Save to YOLO .txt file
        yolo_filename = os.path.join(yolo_txt_path, xml_file.replace(".xml", ".txt"))
        with open(yolo_filename, "w") as f:
            f.write("\n".join(yolo_data))


def split_dataset(lego_dataset_path):
    # Paths
    image_path = f"{lego_dataset_path}/images"
    label_path = f"{lego_dataset_path}/labels"
    split_dataset_path = "lego_dataset/split_dataset"
    train_path = f"{split_dataset_path}/train"
    val_path = f"{split_dataset_path}/val"
    test_path = f"{split_dataset_path}/test"

    # Create directories for train, val, test splits
    for path in [train_path, val_path, test_path]:
        os.makedirs(f'{path}/images', exist_ok=True)
        os.makedirs(f'{path}/labels', exist_ok=True)

    # Get all image filenames
    image_files = [f for f in os.listdir(image_path) if f.endswith(".jpg")]
    random.shuffle(image_files)

    # Split the data into training (70%), validation (15%), and testing (15%)
    train_size = int(0.7 * len(image_files))
    val_size = int(0.15 * len(image_files))

    train_images = image_files[:train_size]
    val_images = image_files[train_size:train_size + val_size]
    test_images = image_files[train_size + val_size:]

    # Copy files
    copy_files(train_images, image_path, label_path, train_path, train_path)
    copy_files(val_images, image_path, label_path, val_path, val_path)
    copy_files(test_images, image_path, label_path, test_path, test_path)

    # Create a YAML file for YOLO training
    create_yaml_file(train_path, val_path, test_path)


# Helper function to copy image and label to the destination folder
def copy_files(file_list, src_image_dir, src_label_dir, dest_image_dir, dest_label_dir):
    for img in tqdm(file_list, desc=f"Copying to {dest_image_dir.split('/')[-1]}", unit="file"):
        shutil.copy(f"{src_image_dir}/{img}", f"{dest_image_dir}/images/{img}")
        label_file = img.replace(".jpg", ".txt")
        shutil.copy(f"{src_label_dir}/{label_file}", f"{dest_label_dir}/labels/{label_file}")


# Create a YAML file for YOLO training
def create_yaml_file(train_path, val_path, test_path):
    # Define the YAML data structure
    yaml_data = {
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'nc': 1,  # Only one class 'lego'
        'names': ['lego']  # Class name 'lego'
    }

    yaml_file = 'lego.yaml'

    # Write the YAML file
    with open(yaml_file, 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)

    print(f"YAML file created at: {os.path.abspath(yaml_file)}")


def main():
    lego_dataset_path = "lego_dataset"

    # Reduce the dataset to 1000 images, convert to single label and copy to a new directory
    reduce_dataset(1000, lego_dataset_path)

    # Convert PASCAL VOC (XML) annotation files to YOLO format txt files
    convert_xml_to_yolo_format(lego_dataset_path)

    # Split the dataset into training, validation, and testing sets
    # Create a YAML file for YOLO training
    split_dataset(lego_dataset_path)


if __name__ == "__main__":
    main()
