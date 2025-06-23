import torch
import json
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, balanced_accuracy_score
import matplotlib.pyplot as plt

def save_confusion_matrix(y_true, y_pred, epoch, split, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix {split} Epoch {epoch+1}')
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{split}_epoch_{epoch+1}.png"))
    plt.close()

def train_moment(model, train_loader, val_loader, class_weights, val_dataset, params, output_dir, update_status):
    import torch._dynamo
    torch._dynamo.disable()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "embeddings"), exist_ok=True)

    with open(os.path.join(output_dir, "params_used.json"), "w") as f:
        json.dump(params, f, indent=2)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(next(model.parameters()).device))
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    device = next(model.parameters()).device

    best_val_acc = 0.0
    best_val_loss = float('inf')
    history = []

    for epoch in range(params["epochs"]):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_y_true, train_y_pred = [], []

        for data, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['epochs']} [Train]"):
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
        save_confusion_matrix(train_y_true, train_y_pred, epoch, "train", os.path.join(output_dir, "metrics"))

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_y_true, val_y_pred = [], []

        with torch.no_grad():
            for data, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{params['epochs']} [Val]"):
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
        save_confusion_matrix(val_y_true, val_y_pred, epoch, "val", os.path.join(output_dir, "metrics"))

        # torch.save(model.state_dict(), os.path.join(output_dir, "models", f"epoch_{epoch+1}.pt"))
        torch.save(model.state_dict(), os.path.join(output_dir, "models", "last_model.pt"))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "models", f"best_model_epoch_{epoch+1}.pt"))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "models", f"best_loss_model_epoch_{epoch+1}.pt"))

        val_precision = precision_score(val_y_true, val_y_pred, average="macro", zero_division=0)
        val_recall = recall_score(val_y_true, val_y_pred, average="macro", zero_division=0)
        val_f1 = f1_score(val_y_true, val_y_pred, average="macro", zero_division=0)
        val_f1_weighted = f1_score(val_y_true, val_y_pred, average="weighted", zero_division=0)
        val_bal_acc = balanced_accuracy_score(val_y_true, val_y_pred)

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "val_f1_weighted": val_f1_weighted,
            "val_balanced_acc": val_bal_acc
        }
        history.append(epoch_metrics)
        with open(os.path.join(output_dir, "metrics", "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        update_status(f"Epoch {epoch+1}/{params['epochs']} completado: Val Acc {val_acc:.4f}")

    save_embeddings(model, val_dataset, device, os.path.join(output_dir, "embeddings"))

def save_embeddings(model, dataset, device, output_dir):
    all_embeddings, all_labels, all_paths = [], [], []
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()
    with torch.no_grad():
        for data, labels, paths in tqdm(loader, desc="Saving embeddings"):
            data = data.to(device)
            output = model(x_enc=data)
            all_embeddings.append(output.embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_paths.extend(paths)

    embeddings = np.concatenate(all_embeddings)
    labels = np.concatenate(all_labels)
    paths = np.array(all_paths)
    np.savez(os.path.join(output_dir, "val_embeddings.npz"),
             embeddings=embeddings,
             labels=labels,
             paths=paths)
