# Experiment 4: Support Vector Machine (SVM)

## Aim

To implement and analyze the performance of a Support Vector Machine
(SVM) model for classification using Python.

## Overview

A Support Vector Machine is a supervised machine-learning algorithm used
for classification. It finds a decision boundary (hyperplane) that
separates classes while maximizing the margin between the closest data
points, called **support vectors**.

This experiment uses the **Iris dataset** and compares two SVM kernels:

-   **Linear kernel** -- suitable when classes can be separated
    approximately by a straight decision boundary.
-   **RBF (Radial Basis Function) kernel** -- can model non-linear
    decision boundaries.

For visualization, the experiment uses the first two features of the
Iris dataset.

## Algorithm

1.  **Start.**
2.  Load the Iris dataset and select the first two features for
    visualization.
3.  Split the data into training and testing sets.
4.  Standardize the features using `StandardScaler`.
5.  Train an SVM classifier with a linear kernel.
6.  Predict the test labels and calculate the classification accuracy.
7.  Train another SVM classifier with the RBF kernel.
8.  Predict the test labels and calculate the classification accuracy.
9.  Plot the decision boundaries for both models.
10. **Stop.**

## Program
    import numpy as np
    import matplotlib.pyplot as plt
    
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    
    # Load Iris dataset
    iris = load_iris()
    X = iris.data[:, :2]
    y = iris.target
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Linear SVM
    linear_model = SVC(kernel='linear')
    linear_model.fit(X_train, y_train)
    
    linear_pred = linear_model.predict(X_test)
    linear_accuracy = accuracy_score(y_test, linear_pred)
    
    # RBF SVM
    rbf_model = SVC(kernel='rbf')
    rbf_model.fit(X_train, y_train)
    
    rbf_pred = rbf_model.predict(X_test)
    rbf_accuracy = accuracy_score(y_test, rbf_pred)
    
    # Display accuracy
    print("Linear SVM Accuracy:", linear_accuracy)
    print("RBF SVM Accuracy:", rbf_accuracy)
    
    # Decision boundary visualization
    x_min, x_max = X_train[:, 0].min()-0.5, X_train[:, 0].max()+0.5
    y_min, y_max = X_train[:, 1].min()-0.5, X_train[:, 1].max()+0.5
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )
    
    for name, model in [
        ("Linear Kernel", linear_model),
        ("RBF Kernel", rbf_model)
    ]:
        Z = model.predict(
            np.c_[xx.ravel(), yy.ravel()]
        )
        Z = Z.reshape(xx.shape)
    
        plt.figure(figsize=(7, 5))
        plt.contourf(xx, yy, Z, alpha=0.25)
        plt.scatter(
            X_train[:, 0], X_train[:, 1],
            c=y_train, edgecolors='black'
        )
    
        plt.title("SVM Decision Boundary - " + name)
        plt.xlabel("Standardized Sepal Length")
        plt.ylabel("Standardized Sepal Width")
        plt.show()
    

## output

    SVM Classification Accuracy
    
    Linear Kernel: 0.7333 (73.33%)

    RBF Kernel: 0.7333 (73.33%)

## Result
Thus, the Support Vector Machine (SVM) classification algorithm was successfully implemented using Python. Linear and RBF kernels were trained and tested using the Iris dataset, and their decision boundaries were visualized. Both models achieved 73.33% accuracy
