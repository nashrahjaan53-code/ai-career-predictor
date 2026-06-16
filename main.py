import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

print("📦 Generating built-in dataset...")
# FIXED: Explicitly set n_redundant=0 and n_clusters_per_class=1 so the math rules pass!
X_raw, y_raw = make_classification(
    n_samples=1000, 
    n_features=5, 
    n_informative=4, 
    n_redundant=0,
    n_classes=3, 
    n_clusters_per_class=1,
    random_state=42
)

# Map features to realistic tech skills
skill_cols = ["Coding_Score", "Math_Stats_Score", "System_Design_Score", "Product_Strategy", "Communication_Score"]
df = pd.DataFrame(X_raw, columns=skill_cols)

# Map numeric targets to actual career paths
career_map = {0: "Software Engineer", 1: "Data Scientist", 2: "Product Manager"}
df["True_Career"] = [career_map[val] for val in y_raw]


# Sklearn's SelfTrainingClassifier expects unlabeled data points to have a label of -1
rng = np.random.default_rng(42)
random_unlabeled_mask = rng.random(y_raw.shape[0]) < 0.75  # Hide 75% of labels

y_semi = y_raw.copy()
y_semi[random_unlabeled_mask] = -1

df["Semi_Labels"] = [career_map[val] if val != -1 else "Unlabeled" for val in y_semi]

print(f" Dataset Ready!")
print(f"   - Total rows: {len(df)}")
print(f"   - Labeled profiles: {np.sum(y_semi != -1)}")
print(f"   - Unlabeled profiles (marked as -1): {np.sum(y_semi == -1)}\n")

# Split features and our semi-supervised targets for training
X_train, X_test, y_train_semi, _ = train_test_split(
    X_raw, 
    y_semi, 
    test_size=0.2, 
    random_state=42
)

# Separately grab the true labels for the same test split to evaluate performance accurately
_, _, _, y_test_true = train_test_split(
    X_raw, 
    y_raw, 
    test_size=0.2, 
    random_state=42
)
print(" Training Self-Training Classifier (Base: RandomForest)...")
base_model = RandomForestClassifier(n_estimators=50, random_state=42)
self_training_model = SelfTrainingClassifier(base_model, threshold=0.75)

# Train using the training features and the semi-labeled targets
self_training_model.fit(X_train, y_train_semi)


predictions = self_training_model.predict(X_test)

print("\n Model Evaluation Metrics:")
print(f"Accuracy Score: {accuracy_score(y_test_true, predictions):.2%}")
print("\nClassification Report:")
print(classification_report(y_test_true, predictions, target_names=list(career_map.values())))

def predict_my_career(coding, math, system, product, comms):
    features = np.array([[coding, math, system, product, comms]])
    pred_idx = self_training_model.predict(features)[0]
    probabilities = self_training_model.predict_proba(features)[0]
    
    print(f"\n Input Skills -> Coding: {coding}, Math: {math}, System Design: {system}, Product: {product}, Comms: {comms}")
    print(f" Predicted AI Career Path: **{career_map[pred_idx]}**")
    print(f" Confidence Rates -> SE: {probabilities[0]:.1%}, DS: {probabilities[1]:.1%}, PM: {probabilities[2]:.1%}")


predict_my_career(coding=2.5, math=-1.2, system=2.1, product=-0.5, comms=0.5)