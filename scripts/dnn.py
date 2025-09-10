import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from kmer_sampling import kmer_sampling_multiple_files, find_files_to_kmerize




file_names, labels = find_files_to_kmerize(directory="data", prefix = ".fna")
X, y = kmer_sampling_multiple_files(directory="data", file_names=file_names, labels = labels, sample_nr = 200)
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


# Dynamically set input and output sizes
input_size = X.shape[1]
output_size = len(np.unique(y))

# If output_size == 2, use single output for binary classification
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        if output_size == 2:
            self.fc3 = nn.Linear(256, 1)
            self.activation = nn.Sigmoid()
        else:
            self.fc3 = nn.Linear(256, output_size)
            self.activation = nn.Softmax(dim=1)

        self.fc2 = nn.Linear(256, 256)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.activation(out)
        return out

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

optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training loop
epochs = 30
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate on test set
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    # For multiclass, take argmax; for binary, threshold at 0.5
    test_outputs = test_outputs.cpu()
    if output_size == 2:
        pred_indices = (test_outputs.numpy() > 0.5).astype(int).flatten()
        true_indices = y_test_tensor.cpu().numpy().astype(int).flatten()
    else:
        pred_indices = np.argmax(test_outputs.numpy(), axis=1)
        true_indices = y_test_tensor.cpu().numpy().astype(int)

    print("Predicted vs Correct class:")
    correct_predictions = 0
    for pred, true in zip(pred_indices, true_indices):
        print(f"Predicted: {id2label[pred]}, Correct: {id2label[true]}")
        
        if id2label[pred] == id2label[true]:
            correct_predictions += 1

    print(f'Correct / total: {correct_predictions}/{X_test.shape[0]}')
    print(f'Accuracy: {(correct_predictions/X_test.shape[0])*100} %')