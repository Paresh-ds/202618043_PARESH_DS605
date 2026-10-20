# DS605 Lab 7: Image Representation, PCA, Eigenfaces, and Face Recognition

**Student ID : 202618043
project:** 202618043_LAB07  
**Course:** DS605 - Fundamentals of Machine Learning  
**Dataset:** AT&T Database of Faces (ORL)

## Project structure

`text
202618043_LAB07/
├── code/
│   └── 202618043_LAB07.ipynb
└── data/
    ├── s1/
    ├── s2/
    └── ...
```

The dataset contains 40 subject folders (`s1` to `s40`), with 10 grayscale images per subject. Images are 112 x 92 pixels.

## Objective

This lab represents face images as numerical vectors, applies PCA using NumPy linear algebra, visualizes eigenfaces, reconstructs images, and compares face recognition in PCA space with raw-pixel matching and traditional machine-learning classifiers.

## Requirements

- Python 3
- NumPy
- Matplotlib
- Pillow
- scikit-learn
- Jupyter Notebook

Install the Python packages with:

```bash
pip install numpy matplotlib pillow scikit-learn jupyter
```

## Run the notebook

1. Place the ORL dataset folders (`s1`, `s2`, ...) inside `data/`.
2. Open `code/202618043_LAB07.ipynb` in Jupyter.
3. Run the notebook cells from top to bottom.

The notebook reads the data from `../data` relative to the `code/` folder. It creates a reproducible 80/20 stratified train-test split with `random_state=42`, so all 40 subjects appear in both sets.

## Method

1. Read each image as grayscale and inspect its dimensions and intensity range.
2. Flatten every 112 x 92 image into a 10,304-value feature vector and retain its subject label.
3. Split the data into training and test sets while preserving each subject in both sets.
4. Compute the mean face using the training set and center the training and test data.
5. Compute principal directions with NumPy SVD. Compare the result against eigendecomposition of the sample-space covariance matrix.
6. Plot cumulative explained variance, retain the minimum number of components reaching 90%, and visualize the leading eigenfaces.
7. Project images into PCA space, reconstruct sample faces, and measure reconstruction error.
8. Compare nearest-face matching using Euclidean distance on raw pixels and PCA features.
9. Train KNN and RBF SVM classifiers on PCA features and tune component counts and model settings with cross-validation on the training set.

PCA/eigenface calculations use NumPy linear-algebra operations rather than scikit-learn PCA, as required. Scikit-learn is used for the traditional classifier comparison.

## Results

### Data and PCA

- Images: **400**
- Subjects: **40**
- Image size: **112 x 92** pixels (**10,304** values per image)
- Train/test split: **320 / 80** images
- Components needed for at least 90% explained variance: **99**
- Cumulative variance at 99 components: **90.08%**
- SVD/covariance relative eigenvalue error: **2.32 x 10^-15**
- First 10 direction cosine similarities: **1.0** (to displayed precision)
- Reconstruction MSE across training images using 99 components: **154.26**

### Recognition

| Method | Components / setting | Correct | Accuracy |
|---|---:|---:|---:|
| Raw-pixel nearest face | 10,304 pixel values | 1 / 80 | 1.25% |
| PCA nearest face | 99 | 79 / 80 | 98.75% |
| KNN, initial settings | 99 components, k=3 | 65 / 80 | 81.25% |
| RBF SVM, initial settings | 99 components, C=10, gamma=scale | 77 / 80 | 96.25% |
| Tuned KNN | 25 components, k=1 | 76 / 80 | 95.00% |
| Tuned RBF SVM | 25 components, C=100, gamma=0.001 | 79 / 80 | 98.75% |

The PCA nearest-face baseline's one error was test image #9: the true subject was `s40`, and the closest training face was labeled `s5`.

The tuned SVM reached 79/80 test accuracy with 25 components. The 99-component setting is retained as the separate result meeting the assignment's 90%-variance requirement. Test accuracy is based on 80 images, so one image changes the reported accuracy by 1.25 percentage points. Hyperparameters were selected by 5-fold cross-validation on training features. Since the PCA basis was computed on the full training set before cross-validation, the CV scores are approximate; the test set was not used to fit the PCA basis.

### Storage estimate

Each image has 10,304 original pixel values. Coefficients alone use 99 values (104.1x fewer per image) or 25 values (412.2x fewer per image). Including one shared PCA basis, one mean face, and coefficients for all 400 images, the estimated total value counts are:

| PCA components | Total storage as a share of original | Reduction |
|---:|---:|---:|
| 99 | 26.0% | 3.85x smaller |
| 25 | 6.7% | 14.83x smaller |

These storage comparisons count values and assume a common numeric precision; they do not include labels, file-format overhead, or metadata.

## Observations

- The leading eigenfaces capture broad patterns of facial variation across the aligned images.
- Keeping 99 components preserves just over 90% of training variance and produces recognizable, though smoother, reconstructions.
- PCA-space Euclidean matching performed substantially better than raw-pixel matching on this split.
- The tuned RBF SVM achieved the same number of correct test predictions as the PCA nearest-face baseline while using 25 components.
- Results depend on the particular reproducible train-test split and should be interpreted with the small test set in mind.

## Figures and notebook outputs

The notebook contains the explained-variance plot, eigenface visualizations, reconstructed sample faces, and printed recognition and classifier results. Run the notebook to reproduce these outputs.
