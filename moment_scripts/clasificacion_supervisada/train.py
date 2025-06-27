import torch
import json
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def save_confusion_matrix(y_true, y_pred, epoch, split):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix {split} Epoch {epoch+1}')
    os.makedirs("outputs/metrics", exist_ok=True)
    plt.savefig(f"outputs/metrics/confusion_matrix_{split}_epoch_{epoch+1}.png")
    plt.close()

def train_moment(model, train_loader, val_loader, class_weights, val_dataset, epochs=20, lr=1e-4):
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/embeddings", exist_ok=True)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(next(model.parameters()).device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device

    best_val_acc = 0.0
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_y_true, train_y_pred = [], []

        for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            data, labels = data.to(device), labels.to(device)
            output = model(x_enc=data)
            loss = criterion(output.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            preds = output.logits.argmax(1)
            train_correct += (preds == labels).sum().item()
            train_total += data.size(0)

            train_y_true.extend(labels.cpu().numpy())
            train_y_pred.extend(preds.cpu().numpy())

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / train_total
        save_confusion_matrix(train_y_true, train_y_pred, epoch, "train")

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_y_true, val_y_pred = [], []

        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                data, labels = data.to(device), labels.to(device)
                output = model(x_enc=data)
                loss = criterion(output.logits, labels)

                val_loss += loss.item() * data.size(0)
                preds = output.logits.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += data.size(0)

                val_y_true.extend(labels.cpu().numpy())
                val_y_pred.extend(preds.cpu().numpy())

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / val_total
        save_confusion_matrix(val_y_true, val_y_pred, epoch, "val")

        # Guardar modelo
        torch.save(model.state_dict(), f"outputs/models/epoch_{epoch+1}.pt")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "outputs/models/best_model.pt")

        # Guardar métricas
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc
        }
        history.append(epoch_metrics)
        with open("outputs/metrics/history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Train Acc {train_acc:.4f}, Val Loss {avg_val_loss:.4f}, Val Acc {val_acc:.4f}")

    # Guardar embeddings del val set
    save_embeddings(model, val_dataset, device)

def save_embeddings(model, dataset, device):
    all_embeddings, all_labels, all_paths = [], [], []
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()
    with torch.no_grad():
        for data, labels, paths in tqdm(loader, desc="Saving embeddings"):
            data = data.to(device)
            output = model(x_enc=data)
            all_embeddings.append(output.embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_paths.extend(paths)  # Guardamos paths

    embeddings = np.concatenate(all_embeddings)
    labels = np.concatenate(all_labels)
    paths = np.array(all_paths)
    np.savez("outputs/embeddings/val_embeddings.npz",
             embeddings=embeddings,
             labels=labels,
             paths=paths)

