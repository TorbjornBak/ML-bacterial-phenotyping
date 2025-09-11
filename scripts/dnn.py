import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from kmer_sampling import kmer_sampling_multiple_files, find_files_to_kmerize
# Create PyTorch datasets and dataloaders
from torch.utils.data import TensorDataset, DataLoader

#import wandb





file_names, labels = find_files_to_kmerize(directory="data", prefix = ".fna")
X, y = kmer_sampling_multiple_files(directory="data", file_names=file_names, labels = labels)
X = np.stack(X, axis=0)

labels = np.unique(y)

label2id = {label: i for i, label in enumerate(labels) }
id2label = {i : label for i, label in enumerate(labels) }

y = [label2id[l] for l in y]
y = np.array(y, dtype=np.float32)
print(y)

# Drop classes with fewer than 2 members
from collections import Counter
class_counts = Counter(y)
valid_classes = [cls for cls, count in class_counts.items() if count >= 2]
mask = np.isin(y, valid_classes)
X_filtered = X[mask]
y_filtered = y[mask]



#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"



if torch.cuda.is_available(): 
    device = torch.device("cuda")

elif torch.backends.mps.is_available(): 
    device = torch.device("mps")
else: 
    device = torch.device("cpu")


print(f"Using {device} device")

print("Splitting data into test and training set")

# Split filtered data into train and test sets (1/5 for test)
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered)


print("Converting arrays to tensors")

# Convert numpy arrays to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device = device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device = device)
y_train_tensor = torch.tensor(y_train, device = device)
y_test_tensor = torch.tensor(y_test, device = device)


class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        layer_size = 512
        self.fc1 = nn.Linear(input_size, layer_size, bias=True)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(layer_size, layer_size, bias=True)
        self.dropout2 = nn.Dropout(0.1)
        if output_size == 2:
            self.fc3 = nn.Linear(layer_size, 1, bias=True)
            self.activation = nn.Sigmoid()
        else:
            self.fc3 = nn.Linear(layer_size, output_size, bias=True)
            self.activation = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        # Gaussian initialization for all Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.activation(out)
        return out
    

learning_rates = [0.001, 0.0001, 0.00001]
epochs = 100
weight_decay = 0.01

for lr in learning_rates:
    # run = wandb.init(
    #     # Set the wandb entity where your project will be logged (generally your team name).
    #     entity="torbjornbak-technical-university-of-denmark",
    #     # Set the wandb project where this run will be logged.
    #     project="Phenotyping bacteria",
    #     # Track hyperparameters and run metadata.
    #     config={
    #         "learning_rate": lr,
    #         "weight_decay" : weight_decay,
    #         "architecture": "NN",
    #         "dataset": "bv-brc.org bacteria class prediction",
    #         "epochs": epochs,
    #     },
    # )

    print(f'Learning rate: {lr}')
    # Cross-validation parameters
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    batch_size = 50

    cv_accuracies = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_filtered, y_filtered)):
        print(f"\nFold {fold+1}/{k_folds}")
        X_train, X_test = X_filtered[train_idx], X_filtered[test_idx]
        y_train, y_test = y_filtered[train_idx], y_filtered[test_idx]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(y_train, device=device)
        y_test_tensor = torch.tensor(y_test, device=device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        input_size = X.shape[1]
        output_size = len(np.unique(y))
        model = SimpleNN(input_size, output_size).to(device)


        # Choose loss function based on output size
        if output_size == 2:
            criterion = nn.BCELoss()
            y_train_tensor = y_train_tensor.float().unsqueeze(1)
            y_test_tensor = y_test_tensor.float().unsqueeze(1)
        else:
            criterion = nn.CrossEntropyLoss()
            y_train_tensor = y_train_tensor.long()
            y_test_tensor = y_test_tensor.long()

        # L2 regularization (weight decay)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Early stopping parameters
        
        patience = 10
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            total_samples = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                if output_size == 2:
                    batch_y = batch_y.float().unsqueeze(1)
                else:
                    batch_y = batch_y.long()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_X.size(0)
                total_samples += batch_X.size(0)
            avg_loss = running_loss / total_samples

            # Validation loss (on test set)
            model.eval()
            val_loss = 0.0
            val_samples = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    if output_size == 2:
                        batch_y = batch_y.float().unsqueeze(1)
                    else:
                        batch_y = batch_y.long()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
                    val_samples += batch_X.size(0)
            avg_val_loss = val_loss / val_samples
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')


            #run.log({"train_loss": avg_loss, "val_loss": avg_val_loss})

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    model.load_state_dict(best_model_state)
                    break

        # Evaluate on test set with DataLoader
        model.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                outputs = outputs.cpu()
                if output_size == 2:
                    preds = (outputs.numpy() > 0.5).astype(int).flatten()
                    trues = batch_y.cpu().numpy().astype(int).flatten()
                else:
                    preds = np.argmax(outputs.numpy(), axis=1)
                    trues = batch_y.cpu().numpy().astype(int)
                all_preds.extend(preds)
                all_trues.extend(trues)

        print("Predicted vs Correct class:")
        correct_predictions = 0
        for pred, true in zip(all_preds, all_trues):
            print(f"Predicted: {id2label[pred]}, Correct: {id2label[true]}")
            if id2label[pred] == id2label[true]:
                correct_predictions += 1
        accuracy = correct_predictions / len(all_trues)
        print(f'Correct / total: {correct_predictions}/{len(all_trues)}')
        print(f'Accuracy: {accuracy*100:.2f} %')
        cv_accuracies.append(accuracy)

    print(f"\nCross-validation mean accuracy: {np.mean(cv_accuracies)*100:.2f} %")




    print("Predicted vs Correct class:")
    correct_predictions = 0
    for pred, true in zip(all_preds, all_trues):
        print(f"Predicted: {id2label[pred]}, Correct: {id2label[true]}")
        if id2label[pred] == id2label[true]:
            correct_predictions += 1
    print(f'Correct / total: {correct_predictions}/{len(all_trues)}')
    print(f'Accuracy: {(correct_predictions/len(all_trues))*100} %')
    #run.finish()