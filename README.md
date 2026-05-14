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


## Business Requirements

The client is a fictional agricultural advisory service working with rice farmers to improve crop health and reduce yield loss caused by leaf diseases. Early disease identification is important for preventing the spread of infections and supporting timely treatment decisions.

Currently, disease detection relies mainly on manual visual inspection, which can be inconsistent, time-consuming, and dependent on expert knowledge. The client is looking for a data-driven solution capable of identifying rice leaf diseases from images.

The project will be considered successful if:
- Visual differences between healthy and diseased rice leaves are identified and presented clearly.
- A machine learning model can accurately classify rice leaf images into healthy or disease categories.
- The results are accessible through an interactive dashboard for non-technical users.

**Business Requirement 1:**
The client is interested in a study that visually differentiates healthy rice leaves from leaves affected by different diseases.

**Business Requirement 2:**
The client is interested in a machine learning solution capable of classifying rice leaf images according to disease type.


## Hypotheses

### Hypothesis 1 - Healthy and Diseased Leaves Have Distinct Visual Patterns

**Hypothesis:**
Healthy rice leaves and diseased rice leaves exhibit noticeable visual differences in color, texture, and lesion patterns.

**Validation:**
- Perform exploratory image analysis on samples from each class.
- Compare visual characteristics between healthy and diseased leaves.
- Analyze image distributions and representative examples.

**Statistical Evidence:**
- Sample image comparisons
- Color distribution analysis
- Class-wise image visualizations


### Hypothesis 2 - Different Rice Diseases Can Be Distinguished from Images

**Hypothesis:**
Different rice leaf diseases contain unique visual characteristics that allow them to be differentiated from one another using image classification techniques.

**Validation:**
- Compare image samples across disease classes.
- Train a classification model and analyze class prediction performance.
- Evaluate which classes are most frequently confused.

**Statistical Evidence:**
- Confusion matrix
- Precision, Recall, and F1-score by class
- Misclassification analysis


### Hypothesis 3 - Class Imbalance Influences Model Performance

**Hypothesis:**
The imbalance in class distribution negatively affects prediction performance for underrepresented disease categories.

**Validation:**
- Analyze dataset class distribution.
- Compare performance metrics across classes.
- Evaluate whether minority classes produce lower prediction accuracy.

**Statistical Evidence:**
- Class distribution plots
- Per-class evaluation metrics
- Comparison of prediction performance between majority and minority classes