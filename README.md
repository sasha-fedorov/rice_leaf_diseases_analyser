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

The client is a fictional agricultural advisor working with rice farmers to improve crop health and yield. Rice diseases represent a major threat to production, often leading to significant losses when not identified and treated early.

Currently, disease detection relies heavily on manual inspection, which is time-consuming, error-prone, and requires expert knowledge that may not always be accessible to farmers in the field. The lack of timely and accurate diagnosis can result in delayed treatment and reduced crop productivity.

The project will be considered successful if:
- Visual differences between healthy and diseased rice leaves are identified and clearly presented.
- A machine learning model can reliably classify rice leaf diseases from images.
- The results are accessible through an interactive dashboard for end users.


### Business requirement 1:
- The client is interested in a study that visually differentiates healthy rice leaves from leaves affected by different diseases.

### Business requirement 2:
- The client is interested in determining whether a given rice leaf image contains a disease and, if so, identifying the specific type of disease.


## Hypotheses

### Hypothesis 1 - Visual Distinction Between Healthy and Diseased Leaves
**Hypothesis:**
There are observable visual differences (such as color, texture, and lesion patterns) between healthy rice leaves and diseased rice leaves.

**Validation:**
- Perform exploratory data analysis (EDA) on image samples from each class.
- Compare color distributions, texture patterns, and visible lesions.
- Visualize class-wise image samples and feature differences.

**Statistical Evidence:**
- Distribution of pixel intensity and color channels across classes
- Image feature comparisons (e.g., mean/variance of RGB values)
- Visual confirmation through plotted samples

### Hypothesis 2 - Distinguishability Between Disease Types
**Hypothesis:**
Different rice leaf diseases exhibit distinct visual patterns that allow them to be differentiated from one another.

**Validation:**
- Compare images across disease classes.
- Analyze intra-class vs inter-class variability.
- Evaluate model confusion matrix to assess misclassification patterns.

**Statistical Evidence:**
- Confusion matrix showing class separability
- Class-wise precision and recall scores
- Feature distribution differences across disease categories

### Hypothesis 3 - Model Can Accurately Classify Rice Leaf Diseases
**Hypothesis:**
A machine learning model can learn to accurately classify rice leaf images into healthy or specific disease categories.

**Validation:**
- Train an image classification model (e.g., CNN or transfer learning).
- Evaluate performance on validation/test data.

**Statistical Evidence:**
- Accuracy score
- Precision, Recall, F1-score
- Loss and accuracy curves during training 

### Hypothesis 4 - Class Imbalance Affects Model Performance
**Hypothesis:**
Imbalanced class distribution negatively impacts the model’s ability to correctly classify underrepresented diseases.

**Validation:**
- Analyze class distribution in the dataset.
- Compare model performance across classes.
- Optionally apply balancing techniques (augmentation, weighting) and re-evaluate.

**Statistical Evidence:**
- Class frequency distribution
- Per-class performance metrics
- Improvement after applying balancing techniques (if implemented)