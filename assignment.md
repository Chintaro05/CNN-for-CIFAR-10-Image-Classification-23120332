# Lab Assignment: CNN for CIFAR-10 Image Classification

## 1. Objective

In this extension lab, you will move beyond MNIST and build a **Convolutional Neural Network (CNN)** for the **CIFAR-10** dataset. Unlike MNIST (28x28 grayscale digits), CIFAR-10 contains **32x32 RGB natural images** across 10 object classes, making it significantly more challenging.

You will:

1. Build a baseline fully connected model for CIFAR-10.
2. Build a CNN model and compare performance.
3. Improve generalization using regularization and data augmentation.
4. Analyze class-wise errors and propose architecture improvements.

---

## 2. Why CIFAR-10 Is Challenging

- Images are color (3 channels), not grayscale.
- Classes are visually more diverse (e.g., cat vs dog, automobile vs truck).
- Objects appear with different backgrounds, poses, and scales.
- Decision boundaries are less linearly separable.

Expected result: a simple MLP that works well on MNIST will perform much worse on CIFAR-10, while CNNs should provide a clear gain.

---

## 3. Dataset and Setup

Use pre-processed CIFAR-10 dataset, `data/cifar10.npz` (inside `cifar10.7z`). Training and test sets are fixed for consistent comparison.

Install dependencies:

```bash
pip install tensorflow numpy scikit-learn matplotlib
```

> **Note:** TensorFlow 2.15.x is recommended for Python 3.9. For Python 3.10+, TensorFlow 2.16+ should work.

---

## 4. Tasks

### Task 1 - Baseline MLP on CIFAR-10

Implement and train a baseline MLP:

- Input: 32x32x3 image flattened to 3072
- Hidden layers: Dense(512, relu), Dense(256, relu)
- Output: Dense(10, softmax)
- Loss: sparse_categorical_crossentropy
- Optimizer: adam

Record test accuracy, parameter count, and confusion matrix.

### Task 2 - CNN Model

Implement and train a CNN:

- Conv2D(32, 3x3, relu) + BatchNormalization
- Conv2D(32, 3x3, relu) + MaxPooling2D + Dropout(0.25)
- Conv2D(64, 3x3, relu) + BatchNormalization
- Conv2D(64, 3x3, relu) + MaxPooling2D + Dropout(0.25)
- Flatten
- Dense(256, relu) + Dropout(0.5)
- Output Dense(10, softmax)

Compile with `adam` and `sparse_categorical_crossentropy`.

### Task 3 - Data Augmentation

Train the same CNN with real-time augmentation (e.g., random horizontal flip, random translation, small rotation). Compare with Task 2 and discuss overfitting reduction.

### Task 4 - Learning Curves and Error Analysis

For MLP and CNN models:

1. Plot training/validation loss and accuracy curves.
2. Generate confusion matrix and classification report.
3. Identify top 3 most confused class pairs and explain why.

### Task 5 - Model Improvement Challenge (Mini Research)

Try at least **one** improvement and report impact:

- Learning-rate scheduling
- Early stopping + model checkpoint
- Deeper CNN block
- GlobalAveragePooling2D instead of Flatten
- Label smoothing

Describe your hypothesis before training and whether the result supports it.

---

## 5. Submission

Submit the following files in [Student ID].zip:

1. Source code (`.py` files or notebook)
2. Trained model files (`.keras`)
3. Metrics outputs (`.json` or `.md`)
4. Figures (learning curves + confusion matrices)
5. A short report (2-4 pages)

If the zip file is too large, upload it to your own sharing service and submit [Student ID].txt with the URL there.
---

## 6. Report Questions

Answer clearly:

1. How much does CNN improve over MLP on CIFAR-10 in accuracy and F1?
2. Which classes are hardest, and what visual factors may explain this?
3. Did augmentation reduce overfitting? Use evidence from curves.
4. Which single improvement gave the best gain per added complexity?
5. If you had 2x more training time, what would you try next?

---

## 7. Suggested Reference Targets

Typical results with reasonable training settings:

- Baseline MLP: ~45% to 55% test accuracy
- Basic CNN (without augmentation): ~70% to 78%
- CNN + augmentation + callbacks: ~75% to 83%

Exact values may vary by hardware, epochs, and random seed.

---

## 8. Grading Rubric (Extension Assignment)

| Criteria | Points |
|---|---|
| Baseline MLP implementation and evaluation | 15 |
| CNN implementation and evaluation | 20 |
| Data augmentation experiment and comparison | 15 |
| Error analysis quality (confusion + class-pair discussion) | 20 |
| Improvement challenge and experimental reasoning | 15 |
| Report clarity and reproducibility | 15 |
| **Total** | **100** |

---

## 9. Optional Advanced Bonus

Use transfer learning with a lightweight pretrained backbone (for example, MobileNetV2) adapted to 32x32 input. Compare performance and training speed against your custom CNN.
