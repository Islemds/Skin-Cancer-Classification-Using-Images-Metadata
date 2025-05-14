import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm.auto import tqdm
import os
from pathlib import Path
import shutil
import sys

from model_builder import LearnedFeatureFusionModel


sys.path.append(str(Path(__file__).resolve().parent))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0
    all_train_preds, all_train_labels = [], []

    for images, skin_features, labels in dataloader:
        images, skin_features, labels = images.to(device), skin_features.to(device), labels.to(device)

        outputs = model(images, skin_features)
        loss = criterion(outputs, labels)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(outputs, dim=1)
        train_acc += (y_pred_class == labels).sum().item() / len(labels)

        all_train_preds.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())
        all_train_labels.append(labels.cpu().numpy())

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    all_train_preds = np.concatenate(all_train_preds, axis=0)
    all_train_labels = np.concatenate(all_train_labels, axis=0)
    
    train_auc = roc_auc_score(all_train_labels, all_train_preds[:, 1])
    train_precision, train_recall, _ = precision_recall_curve(all_train_labels, all_train_preds[:, 1])
    train_auc_pr = auc(train_recall, train_precision)

    return train_loss, train_acc, train_auc, train_auc_pr
    
def eval(model, dataloader, criterion, device):
    model.eval()
    val_loss, val_acc = 0, 0
    all_val_preds, all_val_labels = [], []

    with torch.inference_mode():
        for images, skin_features, labels in dataloader:
            images, skin_features, labels = images.to(device), skin_features.to(device), labels.to(device)

            outputs = model(images, skin_features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            y_pred_class = torch.argmax(outputs, dim=1)
            val_acc += (y_pred_class == labels).sum().item() / len(labels)

            all_val_preds.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())

    val_loss /= len(dataloader)
    val_acc /= len(dataloader)

    all_val_preds = np.concatenate(all_val_preds, axis=0)
    all_val_labels = np.concatenate(all_val_labels, axis=0)
    
    val_auc = roc_auc_score(all_val_labels, all_val_preds[:, 1])
    val_precision, val_recall, _ = precision_recall_curve(all_val_labels, all_val_preds[:, 1])
    val_auc_pr = auc(val_recall, val_precision)

    return val_loss, val_acc, val_auc, val_auc_pr   

def cross_validate_model(
    dataset,
    num_classes=2,
    epochs=1,
    batch_size=8,
    learning_rate=1e-3,
    n_splits=10,
    fusion_operation="mul",
    device=device,
):
    results = {
        "Fold": [],
        "BestFold": [],
        "Epoch": [],
        "TrainLoss": [],
        "TrainAccuracy": [],
        "TrainAUC": [],
        "TrainAUC-PR": [], 
        "TestLoss": [],
        "TestAccuracy": [],
        "TestAUC": [],
        "TestAUC-PR": [], 
    }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)


    fold_models = []
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\nFold {fold + 1}/{n_splits}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True)

        model = LearnedFeatureFusionModel(num_classes=num_classes, fusion_operation=fusion_operation)
        class_weights = torch.tensor([1.33, 4.0]).to(device)
        loss_fn = CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        model.to(device)

        optimizer = Adam(model.parameters(), lr=learning_rate)

        # Initialize lists to store metrics for all epochs
        epoch_train_losses, epoch_train_accuracies, epoch_train_aucs, epoch_train_aucs_pr = [], [], [], []
        epoch_val_losses, epoch_val_accuracies, epoch_val_aucs, epoch_val_aucs_pr = [], [], [], []

        for epoch in range(epochs):
            train_loss, train_acc, train_auc, train_auc_pr = train(
                model, train_dataloader, loss_fn, optimizer, device
            )
            
            val_loss, val_acc, val_auc, val_auc_pr = eval(
                model, val_dataloader, loss_fn, device
            )

            # Append metrics for this epoch
            epoch_train_losses.append(train_loss)
            epoch_train_accuracies.append(train_acc)
            epoch_train_aucs.append(train_auc)
            epoch_train_aucs_pr.append(train_auc_pr)

            epoch_val_losses.append(val_loss)
            epoch_val_accuracies.append(val_acc)
            epoch_val_aucs.append(val_auc)
            epoch_val_aucs_pr.append(val_auc_pr)

            print(f"Epoch {epoch+1} | {epochs}:")
            print(
                f"| Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | Train AUC: {train_auc:.4f} | Train AUC-PR: {train_auc_pr:.4f}"
            )
            print(
                f"| Val Loss: {val_loss:.4f}   | Val Accuracy: {val_acc:.4f}   | Val AUC: {val_auc:.4f}   | Val AUC-PR: {val_auc_pr:.4f}"
            )
            
            results["Fold"].append(fold)
            results["Epoch"].append(epoch)
            results["TrainLoss"].append(train_loss)
            results["TrainAccuracy"].append(train_acc)
            results["TrainAUC"].append(train_auc)
            results["TrainAUC-PR"].append(train_auc_pr)
            results["TestLoss"].append(val_loss)
            results["TestAccuracy"].append(val_acc)
            results["TestAUC"].append(val_auc)
            results["TestAUC-PR"].append(val_auc_pr)
                
                
            results["BestFold"].append(None)

        # Compute averages for the fold
        avg_train_loss = np.mean(epoch_train_losses)
        avg_train_acc = np.mean(epoch_train_accuracies)
        avg_train_auc = np.mean(epoch_train_aucs)
        avg_train_auc_pr = np.mean(epoch_train_aucs_pr)

        avg_val_loss = np.mean(epoch_val_losses)
        avg_val_acc = np.mean(epoch_val_accuracies)
        avg_val_auc = np.mean(epoch_val_aucs)
        avg_val_auc_pr = np.mean(epoch_val_aucs_pr)

        

        fold_models.append(model)
        fold_aucs.append(avg_val_auc)

    print("\nCross-validation complete.")

    best_fold = np.argmax(fold_aucs)
    best_model = fold_models[best_fold]
    best_auc = fold_aucs[best_fold]

    print(f"Best validation AUC: {best_auc:.4f} (Fold {best_fold + 1})")
    results["BestFold"] = [best_fold] * len(results["Fold"])
    

    return best_model, results


def retrain_on_full_data(
    best_model, 
    dataset,
    num_classes=2, 
    epochs=1, 
    batch_size=8, 
    learning_rate=1e-5, 
    patience=5,
    device=device
):
    """
    Train the best model on the full dataset 
    """
    print("\nRetraining the best model on the full dataset...")
    
    full_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    optimizer = Adam(best_model.parameters(), lr=learning_rate)
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)

    best_model.to(device)
    best_model.train()
 
    # Variables for early stopping
    best_loss = float('inf')
    best_auc = -float('inf')
    best_auc_pr = -float('inf')
    patience_counter = 0

    metrics = {
        "Epoch": [],
        "Loss": [],
        "Accuracy": [],
        "AUC": [],
        "AUC-PR": []
    }

       
    for epoch in range(epochs):
        train_loss, train_acc, auc_score, train_auc_pr = train(
        best_model, full_dataloader, loss_fn, optimizer, device
        )
        
        # epoch_loss = train_loss / len(full_dataloader)
        # epoch_acc = train_acc / len(full_dataloader)
        metrics["Epoch"].append(epoch + 1)
        metrics["Loss"].append(train_loss)
        metrics["Accuracy"].append(train_acc)
        metrics["AUC"].append(auc_score)
        metrics["AUC-PR"].append(train_auc_pr)
        
        print(f"Epoch {epoch+1} - Loss: {train_loss:.4f} - Accuracy: {train_acc:.4f} - AUC: {auc_score:.4f} - AUC-PR: {train_auc_pr:.4f}")
        
        
        # Check for improvement
        loss_improved = train_loss < best_loss
        auc_improved = auc_score > best_auc
        auc_pr_improved = train_auc_pr > best_auc_pr
        
        # Update the best metrics
        if loss_improved:
            best_loss = train_loss
        if auc_improved:
            best_auc = auc_score
        if auc_pr_improved:
            best_auc_pr = train_auc_pr
            
        if not (loss_improved or auc_improved or auc_pr_improved):
            patience_counter += 1
        else:
            patience_counter = 0
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} out of {epochs} epochs.")
            break
    return best_model, metrics
