import torch
from torchvision import models, transforms
from PIL import Image
import os

data_dir = "raw_datasets/garfield"
output_dir = "processed_datasets/garfield_cropped"
os.makedirs(output_dir, exist_ok=True)

# Load pre-trained Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Transform for the model
transform = transforms.Compose([
    transforms.ToTensor()
])

# Cat class index in COCO dataset
CAT_CLASS_INDEX = 17

# Iterate through images
for filename in os.listdir(data_dir):
    img_path = os.path.join(data_dir, filename)
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        continue

    # Transform and run inference
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)

    # Filter for cat detections with high confidence
    boxes = outputs[0]['boxes']
    labels = outputs[0]['labels']
    scores = outputs[0]['scores']

    # Find the highest-confidence detection for a cat
    for i, label in enumerate(labels):
        if label == CAT_CLASS_INDEX and scores[i] > 0.8:  # Adjust confidence threshold
            x1, y1, x2, y2 = map(int, boxes[i].tolist())
            cropped_img = img.crop((x1, y1, x2, y2))
            cropped_img = cropped_img.resize((128, 128))  # Resize to 128x128

            # Save the cropped image
            save_path = os.path.join(output_dir, filename)
            cropped_img.save(save_path)
            print(f"Cropped and saved {filename}")
            break
    else:
        print(f"No confident cat detection in {filename}. Skipping.")
