# Rice Leaf Diseases Analyser


## Dataset Content

The dataset consists of annotated images of rice leaves designed for object detection tasks in precision agriculture. It includes both healthy leaves and multiple common rice diseases, enabling the development of computer vision models for automated disease identification.

The data is pre-processed and provided in YOLO format, making it ready for direct use in modern object detection frameworks without additional preparation.

### Dataset Overview

- Total Classes: 9
- Task Type: Object Detection 
- Annotation Format: YOLO (bounding boxes with class IDs)
- Data Split: Training and Validation sets

### Class Labels

The dataset contains the following categories:

| ID | Class Name                     | Description                          |
|----|--------------------------------|--------------------------------------|
| 0  | Bacterial Leaf Blight          | Bacterial disease causing leaf drying |
| 1  | Brown Spot                     | Fungal disease with brown lesions     |
| 2  | Healthy                        | Normal, unaffected rice leaves        |
| 3  | Hispa                          | Pest damage caused by rice hispa      |
| 4  | Leaf Blast                     | Severe fungal infection               |
| 5  | Leaf Scald                     | Disease causing drying leaf edges     |
| 6  | Leaf Smut                      | Fungal disease affecting grains/leaves|
| 7  | Narrow Brown Leaf Spot         | Narrow brown lesions on leaves        |
| 8  | Neck Blast                     | Infection affecting the plant neck    |

### Data Structure

The dataset follows a standard YOLO directory layout:
```
rice/
│
├── images/
│ ├── train/ # Training images
│ └── val/ # Validation images
│
├── labels/
│ ├── train/ # Bounding box annotations
│ └── val/ # Bounding box annotations
│
└── dataset.yaml # Class mapping configuration
```


Each image is paired with a corresponding `.txt` annotation file containing object class IDs and bounding box coordinates.

### Notes

- The dataset is imbalanced, with some classes (e.g., Brown Spot, Leaf Smut) having significantly more samples than others.
- Multiple bounding boxes may exist per image, representing different infected regions.
- Suitable for training deep learning models such as YOLO for real-time disease detection.
