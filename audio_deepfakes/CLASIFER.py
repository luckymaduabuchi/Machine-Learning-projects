import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import os
import librosa
import numpy as np
from torch.nn import Identity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Define the PaSSTFeatureExtractor class
class PaSSTFeatureExtractor(nn.Module):
    def __init__(self, passt_model):
        super(PaSSTFeatureExtractor, self).__init__()
        # Load the PaSST model
        self.passt_model = passt_model
        # Remove the final classification layers
        self.passt_model.head = Identity()
        self.passt_model.head_dist = Identity()

    def forward(self, x):
        # Forward pass through the modified PaSST model
        features = self.passt_model(x)
        return features


# Define a simple classifier with additional layers and batch normalization
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.bn1(torch.relu(self.fc1(x)))
        x = self.bn2(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Define the AudioDataset class for loading audio files
class AudioDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.audio_files = glob.glob(os.path.join(directory, '**', '*.wav'), recursive=True)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        audio = preprocess_audio(audio_path)

        label = None
        if "Real" in audio_path:
            label = 'Real'
        elif "ClonyAI" in audio_path:
            label = 'ClonyAI'
        elif "VoiceAI" in audio_path:
            label = 'VoiceAI'
        elif "VoiceT" in audio_path:
            label = 'VoiceT'
        elif "VoiceChanger" in audio_path:
            label = 'VoiceChanger'
        
        if label is None:
            raise ValueError(f"No label found for the file at path: {audio_path}")

        label_to_int = {'Real': 0, 'ClonyAI': 1, 'VoiceAI': 2, 'VoiceT': 3, 'VoiceChanger': 4}
        label = label_to_int[label]

        return torch.tensor(audio, dtype=torch.float32), label

# Function to preprocess audio
def preprocess_audio(filename):
    audio, sr = librosa.load(filename, sr=32000)  # Load and resample to 32kHz
    if len(audio) > 10 * sr:
        audio = audio[:10 * sr]  # Slice to 5 seconds
    elif len(audio) < 10 * sr:
        audio = np.pad(audio, (0, 10 * sr - len(audio)), "constant")  # Pad with zeros
    return audio

# Define paths to datasets
train_dataset_path = '/home/vm-user/Downloads/Music /Music /train'
val_dataset_path = '/home/vm-user/Downloads/Music /Music /val'

# Create datasets and dataloaders
train_dataset = AudioDataset(train_dataset_path)
val_dataset = AudioDataset(val_dataset_path)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Debugging code to load pre-trained model
model_path = '/home/vm-user/Downloads/Music /Music /model.pth'
try:
    pretrained_model = torch.load(model_path)
    print("Pre-trained model loaded successfully.")
except Exception as e:
    print("Error loading pre-trained model:", e)

# Create the feature extractor using the pre-trained PaSST model
feature_extractor = PaSSTFeatureExtractor(pretrained_model)

# Feature Selection Techniques
# 1. Feature Importance Analysis using Random Forest and Gradient Boosting
def feature_importance_analysis(X, y):
    # Initialize Random Forest classifier
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X, y)

    # Initialize Gradient Boosting classifier
    gb_clf = GradientBoostingClassifier()
    gb_clf.fit(X, y)

    # Get feature importances from Random Forest and Gradient Boosting
    rf_importances = rf_clf.feature_importances_
    gb_importances = gb_clf.feature_importances_

    return rf_importances, gb_importances

# 2. Correlation Analysis
def correlation_analysis(X):
    # Calculate correlation matrix
    corr_matrix = pd.DataFrame(X).corr().abs()

    # Create a mask to identify highly correlated features
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Identify pairs of highly correlated features
    correlated_pairs = [(corr_matrix.columns[i], corr_matrix.columns[j]) for i, j in zip(*np.where(mask)) if i != j]

    return correlated_pairs

# 3. Dimensionality Reduction using PCA
def dimensionality_reduction(X, n_components):
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca

# Extract features and labels from the dataset
X_train = []
y_train = []
for inputs, labels in train_loader:
    features = feature_extractor(inputs).detach().numpy()
    X_train.extend(features)
    y_train.extend(labels.numpy())

X_train = np.array(X_train)
y_train = np.array(y_train)

# Apply Feature Selection
# 1. Feature Importance Analysis
rf_importances, gb_importances = feature_importance_analysis(X_train, y_train)
print("Random Forest Feature Importances:", rf_importances)
print("Gradient Boosting Feature Importances:", gb_importances)

# 2. Correlation Analysis
correlated_pairs = correlation_analysis(X_train)
print("Highly Correlated Feature Pairs:", correlated_pairs)

# 3. Dimensionality Reduction using PCA
X_train_pca = dimensionality_reduction(X_train, n_components=20)

# Define the classifier using the selected features
num_classes = 5
classifier = Classifier(20, num_classes)  # Input dimension is based on the number of PCA components

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# Training loop
train_loss_history = []
train_accuracy_history = []
train_precision_history = []
train_recall_history = []
train_f1_score_history = []
for epoch in range(60):  # Number of epochs
    classifier.train()
    running_loss = 0.0
    total = 0
    correct = 0
    true_labels_train = []
    predicted_labels_train = []
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        # Forward pass through the feature extractor
        features = feature_extractor(inputs)

        # Forward pass through the classifier
        outputs = classifier(features)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        true_labels_train.extend(labels.cpu().numpy())
        predicted_labels_train.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    train_loss_history.append(epoch_loss)
    train_accuracy_history.append(epoch_accuracy)

    epoch_precision = precision_score(true_labels_train, predicted_labels_train, average='macro')
    epoch_recall = recall_score(true_labels_train, predicted_labels_train, average='macro')
    epoch_f1_score = f1_score(true_labels_train, predicted_labels_train, average='macro')

    train_precision_history.append(epoch_precision)
    train_recall_history.append(epoch_recall)
    train_f1_score_history.append(epoch_f1_score)

    print(f'Epoch [{epoch + 1}/{10}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1 Score: {epoch_f1_score:.4f}')

# Validation loop
val_accuracy_history = []
val_precision_history = []
val_recall_history = []
val_f1_score_history = []
for epoch in range(60):  # Number of epochs
    classifier.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    true_labels_val = []
    predicted_labels_val = []
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_features = feature_extractor(val_inputs)
            val_outputs = classifier(val_features)
            val_loss += criterion(val_outputs, val_labels).item()

            _, val_predicted = torch.max(val_outputs, 1)
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

            true_labels_val.extend(val_labels.cpu().numpy())
            predicted_labels_val.extend(val_predicted.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        val_accuracy_history.append(val_accuracy)

        val_precision = precision_score(true_labels_val, predicted_labels_val, average='macro')
        val_recall = recall_score(true_labels_val, predicted_labels_val, average='macro')
        val_f1_score = f1_score(true_labels_val, predicted_labels_val, average='macro')

        val_precision_history.append(val_precision)
        val_recall_history.append(val_recall)
        val_f1_score_history.append(val_f1_score)

        print(f'Epoch [{epoch + 1}/{10}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}, Validation F1 Score: {val_f1_score:.4f}')

# Plotting training loss and accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accuracy_history, label='Training Accuracy')
plt.plot(val_accuracy_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
