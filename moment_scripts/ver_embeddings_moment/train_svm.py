import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

def train_svm(train_embeddings, train_labels, test_embeddings, test_labels, 
              train_files, test_files, class_names, save_dir, save_prefix):
    os.makedirs(save_dir, exist_ok=True)

    clf = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
    clf.fit(train_embeddings, train_labels)

    y_pred_train = clf.predict(train_embeddings)
    y_pred_test = clf.predict(test_embeddings)

    acc_train = accuracy_score(train_labels, y_pred_train)
    acc_test = accuracy_score(test_labels, y_pred_test)

    report_train = classification_report(train_labels, y_pred_train, target_names=class_names, output_dict=True)
    report_test = classification_report(test_labels, y_pred_test, target_names=class_names, output_dict=True)
    cm_test = confusion_matrix(test_labels, y_pred_test)

    results = {
        "train_accuracy": acc_train,
        "test_accuracy": acc_test,
        "train_report": report_train,
        "test_report": report_test
    }
    with open(os.path.join(save_dir, f"{save_prefix}.json"), "w") as f:
        json.dump(results, f, indent=4)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Test Confusion Matrix")
    plt.savefig(os.path.join(save_dir, f"{save_prefix}_cm.png"))
    plt.close()

    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(test_embeddings)
    np.savez(os.path.join(save_dir, f"{save_prefix}_embeddings.npz"),
             test_embeddings=test_embeddings,
             test_labels=test_labels,
             test_files=test_files,
             train_embeddings=train_embeddings,
             train_labels=train_labels,
             train_files=train_files)

    print(f"Métricas y embeddings guardados en {save_dir}")
