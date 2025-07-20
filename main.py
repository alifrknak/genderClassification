import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score

# 1. Load data
df = pd.read_csv('gender.csv')  # make sure data.csv is in your working dir

# 2. Prepare features and target
X = df[['weight', 'length', 'age']]
y = df['gender']

# 2a. Encode labels to integers (e.g. 'Male'/'Female' → 0/1)
le = LabelEncoder()
y_enc = le.fit_transform(y)  

# 2b. (Optional) scale features for better convergencecls
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# 4. Train a classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Predict on test set
y_pred = model.predict(X_test)

# 6. Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le.classes_  
)

# fig, ax = plt.subplots(figsize=(6, 6))
# disp.plot(ax=ax, cmap='Blues')
# plt.title('Confusion Matrix for Gender Classification')
# plt.show()

print(classification_report(y_test, y_pred, target_names=le.classes_))

# 1. Accuracy
accuracy = accuracy_score(y_test, y_pred)

# 2. Precision (positive predictive value)
precision = precision_score(y_test, y_pred)

# 3. Recall (true positive rate)
recall = recall_score(y_test, y_pred)

# 4. Confusion Matrix Elements
tn, fp, fn, tp = cm.ravel()  # cm must be 2x2

# 5. False Positive Rate (FPR = FP / (FP + TN))
fpr = fp / (fp + tn)

# 6. Print them
print(f"Accuracy : {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"FPR      : {fpr:.2f}")