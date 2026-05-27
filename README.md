# Rice Leaf Diseases Analyser


## Dataset Content

The dataset consists of annotated images of rice leaves for disease classification in precision agriculture. It includes healthy leaves and multiple common rice diseases, enabling the development of computer vision models for automated disease identification and disease type classification.

**Dataset Format & Usage Note:** The data is stored in YOLO format (bounding boxes with class IDs), but it is used functionally in this project as a single-label image classification dataset. Each image contains a single disease category or healthy status. The bounding boxes provide spatial context in the source annotations but are not consumed by the current classification pipeline.

**Class Mapping Location:** Class-to-ID mapping is defined inside the project notebooks (preprocessing/modeling) based on `data.yaml` from dataset.

### Dataset Overview

- Total Classes: 9 (1 healthy + 8 disease types)
- Primary Task: Multi-class Disease Classification
- Storage Format: YOLO (bounding box coordinates with class IDs)
- Functional Usage: Single-label image classification
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
└── data.yaml # Class mapping configuration
```


Each image is paired with a corresponding `.txt` annotation file containing class IDs and bounding box coordinates of the affected region.

### Dataset Usage Notes

- **Single Label per Image:** Each image contains exactly one disease class label, representing either a healthy leaf or a single disease type affecting the leaf.
- **Class Imbalance:** The dataset is imbalanced, with some classes (e.g., Brown Spot, Leaf Smut) having significantly more samples than others. This imbalance is reflected in model performance across classes.
- **Bounding Box Use:** Bounding box coordinates indicate the affected area on the leaf, providing spatial context for the disease manifestation.
- **Current Task:** Images are classified into one of 9 categories (healthy or one of 8 disease types), making this a multi-class image classification problem rather than an object detection problem.


## Business Context & Requirements

The client is a fictional agricultural advisory service working with rice farmers to improve crop health and reduce yield loss caused by leaf diseases. Early and accurate disease identification is critical for:
- Preventing the spread of infections through timely intervention
- Supporting treatment decisions specific to the identified disease
- Reducing crop losses from undiagnosed or misdiagnosed diseases

**Current Challenge:**
Disease detection relies mainly on manual visual inspection by farmers or extension officers, which is:
- Inconsistent across different inspectors and growing regions
- Time-consuming for large-scale farm monitoring
- Dependent on expert knowledge that may not be widely available
- Prone to human error, especially for visually similar diseases

**Business Requirements:**

**Requirement 1 - Visual Evidence & Analysis:**
The client is interested in a study that visually differentiates healthy rice leaves from leaves affected by different diseases. This supports farmer education and builds confidence in automated recommendations.

**Requirement 2 - Automated Disease Classification:**
The client is interested in a machine learning solution capable of accurately classifying rice leaf images to identify both:
- Whether the leaf is healthy or infected
- If infected, the specific disease type (from 8 disease categories)

This classification enables targeted, disease-specific treatment recommendations.


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


## Machine Learning Business Case

### Problem Statement
Rice farmers need a fast, accurate, and accessible method to identify and classify rice leaf diseases from leaf images, enabling timely and disease-specific treatment decisions.

### Learning Approach
**Multi-class Image Classification using Convolutional Neural Networks (CNN)**

- **Why Classification:** Each leaf image shows a single dominant disease state (healthy or one disease type). The goal is to assign it to the most likely category.
- **Why CNN:** Convolutional neural networks are well-suited for learning spatial patterns in images (colors, textures, lesion shapes) that distinguish disease types.
- **Model Architecture:** Transfer learning approach using pre-trained MobileNetV2 backbone, fine-tuned on rice leaf disease data. This approach balances model complexity with training efficiency.

### Ideal Outcome
A model that can:
1. **Correctly identify healthy leaves** - Minimize false positives (healthy leaves misclassified as diseased) to avoid unnecessary treatment costs.
2. **Accurately classify disease type** - For infected leaves, predict the correct disease with high confidence to enable targeted treatment.
3. **Handle visual ambiguity** - When diseases are visually similar, make confident predictions based on learned patterns.
4. **Perform consistently** - Maintain accuracy across all disease classes despite class imbalance in training data.

### Success Metrics (Model Performance Requirements)

**Overall Model Performance:**
- **Validation Accuracy ≥ 75%:** Model correctly classifies at least 75% of validation images across all 9 classes.
- **Macro-averaged F1-score ≥ 0.70:** Average precision and recall across all disease classes is satisfactory, indicating balanced performance even for minority classes.

**Per-Class Performance:**
- **Healthy Class Precision ≥ 80%:** When the model predicts "Healthy," it should be correct at least 80% of the time (minimize false positives).
- **Disease Classes Recall ≥ 65%:** For each disease class, the model should identify at least 65% of actual disease cases (minimize missed infections).
- **No Class Below 50% Accuracy:** Even minority disease classes should achieve >50% accuracy to be actionable.

**Business Applicability:**
- **Dashboard Usability:** Model predictions are presented with confidence scores and reasoning on an interactive dashboard.
- **Dashboard Usability:** Model predictions are presented with confidence scores and reasoning on an interactive dashboard.
- **Farmer Trust:** The dashboard presents sample images and prediction explanations.
- **Actionable Recommendations:** Each prediction includes suggested treatment and next steps.

### Model Output & User Relevance

**What the Model Provides:**
- **Disease Classification:** Prediction of one class from 9 categories (Healthy or 8 disease types).
- **Confidence Score:** Probability distribution across all classes, showing model certainty.
- **Visual Evidence (current status):** The source dataset includes bounding boxes that mark affected regions, but the existing ML pipeline does not produce bounding-box overlays or region highlights. The notebooks and dashboard present model predictions, confidence scores, and example images instead.

**Planned Enhancement:** Add localized visual explanations (e.g., bounding-box overlays, Grad-CAM, or attention maps) in future work to show where the model focuses when making predictions.

**How Farmers Use It:**
- Farmers photograph or upload a rice leaf image to the dashboard.
- Model returns disease classification with confidence.
- Farmer receives disease-specific treatment recommendations and timing guidance.
- Enables faster decision-making compared to manual inspection or waiting for expert consultation.

### Heuristics & Training Data Used

**Data Preprocessing:**
- Images resized to 256×256 pixels for model input.
- Data augmentation applied during training: rotations, flips, brightness/contrast adjustments to improve generalization.
- Class labels extracted from YOLO annotation files; single disease label per image used for classification.

**Training Configuration:**
- **Base Model:** MobileNetV2 pre-trained on ImageNet, transfer learning approach.
- **Loss Function:** Categorical cross-entropy (appropriate for multi-class classification).
- **Optimizer:** Adam optimizer with default learning rates.
- **Early Stopping:** Training stops if validation loss does not improve for a patience period, preventing overfitting.
- **Class Weighting:** Applied to handle dataset imbalance—minority disease classes receive higher loss weights during training.

### Known Limitations & Considerations

1. **Dataset Imbalance:** Some diseases (e.g., Leaf Smut, Brown Spot) have many more training examples than others (e.g., Neck Blast). This may result in lower accuracy for minority classes.
2. **Single-Image Classification:** Each prediction is based on a single leaf image. Farmer expertise and field context should also inform final treatment decisions.
3. **Environmental & Cultivar Variations:** The dataset may not cover all rice cultivars, growing regions, or environmental conditions. Model performance may vary in new contexts.
4. **Bounding Box Information:** While bounding boxes indicate affected regions, the current model uses the full image for classification, not spatial attention to the affected area.

### Next Steps for Improvement

- Collect more data for minority disease classes to improve balance.
- Explore class-balanced sampling or advanced imbalance handling techniques.
- Investigate misclassified examples to understand visual confusion between disease types.
- Consider ensemble methods combining multiple models for improved robustness.
- Integrate farmer feedback to refine recommendations and model retraining strategy.