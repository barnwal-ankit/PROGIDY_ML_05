import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

print("\n--- Food Recognition---")
print("Dataset: https://www.kaggle.com/dansbecker/food-101\n")


num_samples = 1000
image_size = 100 * 100 
num_classes = 10
X_dummy = np.random.rand(num_samples, image_size) + np.random.normal(0, 0.05, (num_samples, image_size))
# Simulate labels for 10 classes
y_dummy = np.random.randint(0, num_classes, num_samples)
labels = [f'Food_{i}' for i in range(num_classes)]


X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size=0.25, random_state=42, stratify=y_dummy)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg_model = LogisticRegression(solver='saga', multi_class='multinomial', random_state=42, max_iter=200, C=0.1, n_jobs=-1) # Use saga for speed, lower C for regularization
print(f"Training Logistic Regression model ({num_classes} classes)...")
logreg_model.fit(X_train_scaled, y_train)
print("Training complete.")

y_pred = logreg_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=labels, zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nModel Evaluation (on dummy data):")
print(f"  Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(conf_matrix)


if num_classes <= 15: 
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - LogReg Food Rec ({num_classes} classes - Dummy Data)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    print(f"\nShowing Confusion Matrix Visualization ({num_classes} classes)...")
    plt.show()
else:
    print("\nConfusion Matrix visualization skipped (too many classes for clear plot).")
