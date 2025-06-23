import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def train_svm(train_embeddings, train_labels, test_embeddings, test_labels, class_names, save_dir, save_prefix):
    """
    Entrena SVM, evalúa, y guarda métricas, gráficos, modelo y embeddings.
    """
    os.makedirs(save_dir, exist_ok=True)

    clf = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
    clf.fit(train_embeddings, train_labels)

    # Train
    y_pred_train = clf.predict(train_embeddings)
    acc_train = accuracy_score(train_labels, y_pred_train)
    report_train = classification_report(train_labels, y_pred_train, target_names=class_names, output_dict=True)
    cm_train = confusion_matrix(train_labels, y_pred_train)

    # Test
    y_pred_test = clf.predict(test_embeddings)
    acc_test = accuracy_score(test_labels, y_pred_test)
    report_test = classification_report(test_labels, y_pred_test, target_names=class_names, output_dict=True)
    cm_test = confusion_matrix(test_labels, y_pred_test)

    # Guardar métricas
    results = {
        "train_accuracy": acc_train,
        "test_accuracy": acc_test,
        "train_report": report_train,
        "test_report": report_test
    }
    json_path = os.path.join(save_dir, f"{save_prefix}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Métricas guardadas en {json_path}")

    # Matriz de confusión
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(save_dir, f"{save_prefix}_cm.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix guardada en {cm_path}")

    # PCA 2D
    pca_2d = PCA(n_components=2)
    emb_pca_2d = pca_2d.fit_transform(test_embeddings)
    plt.figure(figsize=(7,6))
    scatter = plt.scatter(emb_pca_2d[:,0], emb_pca_2d[:,1], c=test_labels, cmap='Set1', alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=class_names)
    plt.title("Test embeddings PCA 2D")
    plt.tight_layout()
    pca2d_path = os.path.join(save_dir, f"{save_prefix}_pca2d.png")
    plt.savefig(pca2d_path)
    plt.close()
    print(f"PCA 2D plot guardado en {pca2d_path}")

    # PCA 3D
    pca_3d = PCA(n_components=3)
    emb_pca_3d = pca_3d.fit_transform(test_embeddings)
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(emb_pca_3d[:,0], emb_pca_3d[:,1], emb_pca_3d[:,2], c=test_labels, cmap='Set1', alpha=0.7)
    ax.set_title("Test embeddings PCA 3D")
    plt.tight_layout()
    pca3d_path = os.path.join(save_dir, f"{save_prefix}_pca3d.png")
    plt.savefig(pca3d_path)
    plt.close()
    print(f"PCA 3D plot guardado en {pca3d_path}")

    # Guardar modelo SVM
    model_path = os.path.join(save_dir, f"{save_prefix}_svm.joblib")
    joblib.dump(clf, model_path)
    print(f"Modelo SVM guardado en {model_path}")

    # Guardar embeddings
    embeddings_path = os.path.join(save_dir, f"{save_prefix}_embeddings.npz")
    np.savez(embeddings_path, train_embeddings=train_embeddings, train_labels=train_labels,
             test_embeddings=test_embeddings, test_labels=test_labels)
    print(f"Embeddings guardados en {embeddings_path}")

    return clf
