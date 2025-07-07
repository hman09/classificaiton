# Breast Cancer Classification (Logistic Regression)

This project uses logistic regression to classify breast cancer tumors as malignant or benign using the `sklearn.datasets.load_breast_cancer()` dataset.

Four models were tested:

* **Model 1**: Raw (Unmanipulated)
* **Model 2**: Standardised (All Features)
* **Model 3**: Feature Selection (Top 10 Correlated Features, No Scaling)
* **Model 4**: Feature Selection + Standardised

---

### --- Model 1: Raw (Unmanipulated)

**Accuracy**: 0.9561

```
Classification Report:
               precision    recall  f1-score   support

           0       0.97      0.91      0.94        43
           1       0.95      0.99      0.97        71

Confusion Matrix:
[[39  4]
 [ 1 70]]
```

---

### --- Model 2: Standardised (All Features)

**Accuracy**: 0.9737

```
Classification Report:
               precision    recall  f1-score   support

           0       0.98      0.95      0.96        43
           1       0.97      0.99      0.98        71

Confusion Matrix:
[[41  2]
 [ 1 70]]
```

---

### --- Model 3: Feature Selection (No Scaling)

**Accuracy**: 0.9912 ✅ Best

```
Classification Report:
               precision    recall  f1-score   support

           0       1.00      0.98      0.99        43
           1       0.99      1.00      0.99        71

Confusion Matrix:
[[42  1]
 [ 0 71]]
```

---

### --- Model 4: Feature Selection + Standardised

**Accuracy**: 0.9737

```
Classification Report:
               precision    recall  f1-score   support

           0       0.95      0.98      0.97        43
           1       0.99      0.97      0.98        71

Confusion Matrix:
[[42  1]
 [ 2 69]]
```

---

### Notes:

* Feature selection was based on the top 10 features most correlated with the target.
* Standardisation was done using `StandardScaler`.

---

### To Run:

```bash
pip install -r requirements.txt
python app.py
```

---

🔪 **Best performance** was achieved with Model 3 (Feature Selection without Scaling).
