# Lab 06: Crack Image Classification

This lab explores basic image preprocessing and binary classification using the crack image dataset. The notebook loads labeled images from the `data/image` directory, converts each image to grayscale, normalizes pixel values, and trains a logistic regression model to classify images as containing a crack or not.

## Project structure

```text
202618043_LAB06/
├── code/
│   └── 202618043_LAB06.ipynb
├── data/
│   ├── image/
│   │   ├── Cracks/       # Images labeled as cracks (class 1)
│   │   └── NonCracks/    # Images labeled as non-cracks (class 0)
│   └── text/
│       └── emails.csv
└── README.md
```

The notebook uses the image folders. The `emails.csv` file is present in the project but is not used by this notebook.

## Requirements

Python 3.9 or later is recommended. Install the libraries used by the notebook:

```bash
pip install numpy pillow matplotlib scikit-learn jupyter
```

## Run the notebook

1. Open a terminal in the `202618043_LAB06` project directory.
2. Start Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open `code/202618043_LAB06.ipynb` and run the cells from top to bottom.

The notebook uses relative data paths such as `../data/image`. Run it from the notebook's `code` directory (or set the notebook server's root so those paths resolve correctly).

## Workflow

1. Load and display a sample crack image.
2. Explore resizing, grayscale conversion, pixel values, normalization to the range 0–1, and flattening into a feature vector.
3. Read images from `data/image/Cracks` and `data/image/NonCracks`. The folder names provide the labels:
   - `Cracks` → `1`
   - `NonCracks` → `0`
4. Convert images to grayscale, normalize their pixel values, and flatten them into feature vectors.
5. Split the data into training and test sets with an 80/20 split, stratified by class, using `random_state=42`.
6. Train a scikit-learn `LogisticRegression` classifier with `max_iter=1000`.
7. Report accuracy, per-class precision/recall/F1, and the confusion matrix; then demonstrate prediction and class probabilities for a sample image.

## Important implementation detail

The notebook currently has the image resize lines commented out during dataset loading and prediction. It therefore expects every image to have the same dimensions so that flattened feature vectors have equal lengths. The supplied image files are named as 448-pixel resized images. The comments referring to `128 × 128` and `16,384` features describe the disabled resize path and do not reflect the current dataset-loading code. If using images with varying dimensions, enable resizing in both dataset loading and `preprocess_image` (and use the same target size in both places).

## Results

The notebook prints metrics when run. Results may depend on the dataset and installed library versions; this README does not assume a fixed accuracy value.
