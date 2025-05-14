# Skin Cancer Classification Using Images & Metadata

This project aims to develop machine learning models that classify skin lesions as benign or malignant by leveraging both dermoscopic images and patient metadata. By integrating visual features with contextual information such as age, sex, and anatomical site, the goal is to enhance diagnostic accuracy — potentially aiding in early detection and reducing unnecessary biopsies.

Skin cancer classification across different skin tones is a complex task, as its appearance can vary significantly depending on the patient’s pigmentation. This project aims to benchmark several image classification models, including those that incorporate additional patient metadata — such as skin tone indicators — and combine it with features extracted from dermoscopic images.


##  Project Overview

Skin cancer, particularly melanoma, poses significant health risks. Early and accurate diagnosis is crucial for effective treatment. This project explores the use of deep learning techniques to automate and improve the classification process by:

- Utilizing dermoscopic images of skin lesions.
- Incorporating patient metadata to provide contextual information.
- Applying data augmentation to enhance model generalization.

## Repository Structure

```
├── Data Processing.ipynb                 # Notebook for initial data exploration and preprocessing
├── ImageOnly/                            # Models trained solely on image data
├── LearnedFeatureModel/                  # Models trained on combined image and metadata features without melama feature
├── LearnedFeatureModelWithMelanoma/      # Models trained on combined image and metadata features with melama feature
└── README.md                             # Project documentation
```

## Dataset

The dataset includes dermoscopic images of skin lesions and patient metadata:

- **Image file**
- **Skin Tone**
- **disease**
- **Malignant**

*The dataset source: [ddi dataset]([https://www.isic-archive.com/](https://ddi-dataset.github.io/))*

## Methodology

1. **Preprocessing**
   - Clean metadata, resize/normalize images
   - Handle missing values and outliers

2. **Data Augmentation**
   - Apply rotation, flipping, scaling to improve model generalization

3. **Model Development**
   - **Image-Only Models** 
   - **Combined Models** (Without the melanoma feature)
   - **Melanoma Models** (With the melanoma feature)

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score, Confusion Matrix

##  Getting Started

### Requirements

- Python 3.6+
- Jupyter Notebook
- Common Python libraries (NumPy, Pandas, PyTorch/TensorFlow, etc.)

### Installation

```bash
git clone https://github.com/Islemds/Skin-Cancer-Classification-Using-Images-Metadata.git
cd Skin-Cancer-Classification-Using-Images-Metadata
```

(Optional) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies (create a `requirements.txt` file if needed):

```bash
pip install -r requirements.txt
```

### Running the Notebooks

```bash
jupyter notebook
```

Then run:
1. `Data Processing.ipynb`

Adjust dataset paths as needed.

## Results

To evaluate model performance:

- Train models using the provided notebooks
- Use evaluation metrics (accuracy, F1, etc.)
- Interpret confusion matrices for class-specific insights

## Future Work

- Use additional metadata (e.g., patient history)
- Deploy as a web app for clinical use

