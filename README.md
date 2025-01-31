# Evaluation of KNN, SVM, and Decision Tree Classifiers on the Cardiotocography Dataset

![Cardiotocography](https://github.com/aidandf29/Classification-of-the-Cardiotocography-Dataset./blob/main/Cardiotocography-CTG.jpg.webp)

## 📝 Project Description

This project aims to evaluate the performance of three different classifiers: K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Decision Tree (DT) in classifying the Cardiotocography (CTG) dataset. The dataset contains 2,126 records categorized into three classes: Normal (N), Suspect (S), and Pathologic (P). The main objective of this project is to compare the accuracy and performance of these classifiers.

## 🚀 Key Features

Evaluation of Three Classifiers: Comparing the performance of KNN, SVM, and Decision Tree.
Hyperparameter Tuning: Utilizing GridSearch to find the best parameters.
High Accuracy: Achieving up to 91% accuracy with Decision Tree.
In-Depth Analysis: Providing detailed insights into the strengths and weaknesses of each classifier.

## 🛠️ Technologies Used

Python: The primary programming language used.
Scikit-learn: Library for implementing KNN, SVM, and Decision Tree.
Pandas: For data manipulation and analysis.
NumPy: For numerical computations.
Matplotlib & Seaborn: For data visualization and result interpretation.

## 📂 Repository Structure

```
/ClassifierEvaluation
├── /code # Jupyter notebooks untuk preprocessing, training, dan evaluasi
├── result # experiment result
├── README.md # This README file
```

## 📊 Results and Evaluation

### KNN Classifier

| Parameter        | Precision | Recall | F1-Score | Accuracy |
| ---------------- | --------- | ------ | -------- | -------- |
| Best Parameter   | N: 0.93   | 0.96   | 0.94     | 0.89     |
|                  | S: 0.66   | 0.59   | 0.62     |          |
|                  | P: 0.91   | 0.77   | 0.83     |          |
| Default          | N: 0.92   | 0.96   | 0.94     | 0.89     |
|                  | S: 0.66   | 0.55   | 0.60     |          |
|                  | P: 0.89   | 0.75   | 0.81     |          |
| Random Variation | N: 0.92   | 0.97   | 0.94     | 0.89     |
|                  | S: 0.68   | 0.55   | 0.61     |          |
|                  | P: 0.93   | 0.77   | 0.84     |          |

### SVM Classifier

| Kernel     | Precision | Recall | F1-Score | Accuracy |
| ---------- | --------- | ------ | -------- | -------- |
| Linear     | N: 0.91   | 0.96   | 0.94     | 0.88     |
|            | S: 0.67   | 0.46   | 0.54     |          |
|            | P: 0.76   | 0.79   | 0.77     |          |
| RBF        | N: 0.87   | 0.97   | 0.92     | 0.84     |
|            | S: 0.51   | 0.33   | 0.40     |          |
|            | P: 0.97   | 0.42   | 0.58     |          |
| Polynomial | N: 0.89   | 0.97   | 0.93     | 0.87     |
|            | S: 0.65   | 0.43   | 0.52     |          |
|            | P: 0.81   | 0.62   | 0.70     |          |

### Decision Tree Classifier

| Parameter        | Precision | Recall | F1-Score | Accuracy |
| ---------------- | --------- | ------ | -------- | -------- |
| Best Parameter   | N: 0.94   | 0.97   | 0.96     | 0.91     |
|                  | S: 0.75   | 0.68   | 0.71     |          |
|                  | P: 0.77   | 0.71   | 0.74     |          |
| Default          | N: 0.94   | 0.95   | 0.95     | 0.91     |
|                  | S: 0.71   | 0.71   | 0.71     |          |
|                  | P: 0.94   | 0.85   | 0.89     |          |
| Random Variation | N: 0.92   | 0.93   | 0.93     | 0.88     |
|                  | S: 0.61   | 0.59   | 0.60     |          |
|                  | P: 0.84   | 0.81   | 0.83     |          |

### Accuracy Comparison

| No  | Variation            | Accuracy |
| --- | -------------------- | -------- |
| 1   | KNN Hyperparameter   | 0.8942   |
| 2   | KNN Default          | 0.8886   |
| 3   | KNN Random Variation | 0.8942   |
| 4   | DT Hyperparameter    | 0.9075   |
| 5   | DT Default           | 0.9103   |
| 6   | DT Random Variation  | 0.8772   |
| 7   | SVM Hyperparameter   | 0.8772   |
| 8   | SVM Default          | 0.8404   |
| 9   | SVM Random Variation | 0.8659   |

### Analysis

#### KNN Classifier

From the results obtained in the KNN Classifier table, the best performance was achieved using hyperparameter tuning with GridSearch. Although the difference between the hyperparameter-tuned and random variation experiments was minimal, with accuracy values being the same up to 10 decimal places, the hyperparameter-tuned approach performed slightly better. This is because the two experiments only differed in two parameters: `n_neighbors` and `algorithm`, which had a less significant impact on the results.

#### Decision Tree Classifier

- In the Decision Tree Classifier experiments, the combination of the `criterion='entropy'` and `max_features='log2'` hyperparameters yielded the best overall accuracy. This was achieved through hyperparameter tuning using GridSearch to find the optimal combination.
- The use of `criterion='entropy'` provided better information gain, allowing the model to more easily identify important features in the data.
- The use of `max_features='log2'` sped up the training process by limiting the number of features used in each split to `log2(23)`.
- Increasing `max_depth` reduced accuracy due to overfitting, as excessively deep trees tend to overfit the training data.

#### SVM Classifier

- Among all the kernels used in the SVM classification, the **linear kernel** achieved the best overall accuracy. This indicates that the dataset is likely **linearly separable**.
- The **RBF kernel** performed poorly because it assumes the data is normally distributed, which was not the case with this dataset.
- The **polynomial kernel** is generally used for image processing and is less suitable for this dataset, as it requires more computational resources and did not perform as well as the linear kernel.

### Conclusion

From the experiments conducted, the **Decision Tree Classifier** achieved the best results. This is because the dataset used in this experiment is well-suited for Decision Tree, which performs well on non-linear data. The dataset is also not continuous, which is a drawback for Decision Tree classifiers when dealing with continuous data.

## 📞 Contact

Muhammad Aidan Daffa Junaidi - muhammad.aidan@ui.ac.id

GitHub: aidandf29
