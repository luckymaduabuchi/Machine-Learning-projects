import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import glob
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from hear21passt.base import get_basic_model, get_model_passt
from hear21passt.models.preprocess import AugmentMelSTFT
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import torch.nn.functional as F

# Then, within your model definition or forward pass, you can


# Configuration for the Mel spectrogram extractor
# Load the basic model structure
model = get_basic_model(mode="logits")

# Load a pre-trained model, example using one trained on AudioSet for 10-second clips
model.net = get_model_passt(arch="passt_s_swa_p16_128_ap476", n_classes=527)  # Example

# Configuration for the Mel spectrogram extractor
model.mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=640, n_fft=1024)


def preprocess_audio(filename):
    audio, sr = librosa.load(filename, sr=32000)  # Load and resample to 32kHz
    if len(audio) > 5 * sr:
        audio = audio[:5 * sr]  # Slice to 3 seconds
    elif len(audio) < 5 * sr:
        audio = np.pad(audio, (0, 5 * sr - len(audio)), "constant")  # Pad with zeros
    return audio


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
    
    
def make_weights_for_balanced_classes(dataset, nclasses):                        
    count = [0] * nclasses                                                      
    for _, label in dataset:                                                        
        count[label] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(dataset)                                              
    for idx in range(len(dataset)):                                            
        _, label = dataset[idx]
        weight[idx] = weight_per_class[label]                                  
    return weight


# Specify the path to your training and validation folders
train_dataset = AudioDataset('/home/vm-user/Downloads/Music /Music /train')
val_dataset = AudioDataset('/home/vm-user/Downloads/Music /Music /val')


weights = make_weights_for_balanced_classes(train_dataset, 5)  # Pass the entire dataset object
weights = torch.DoubleTensor(weights)                                       
sampler = WeightedRandomSampler(weights, len(weights))    

train_loader = DataLoader(train_dataset, batch_size=5, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)

for batch in train_loader:
    audio_wave, labels = batch
    # Pass audio_wave through the model to get features/logits
    features = model(audio_wave)
    print(features.shape)  # Shape of extracted features
    break

class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.fc2 = nn.Linear(input_size // 2, num_classes)

    def forward(self, x):
        # Reshape x to ensure its size matches the expected input size of the linear layer
        x = x.view(-1, self.fc1.in_features)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# Initialize the classifier model with dropout



# Define the necessary hyperparameters
input_size = 527  # Adjusted to match the size of the extracted features
num_classes = 5  # Number of classes
learning_rate = 0.0001
num_epochs = 100


classifier_model = Classifier(input_size, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier_model.parameters(), lr=learning_rate)

train_losses = []  # To store average loss per epoch
train_accuracies = []  # To store accuracy per epoch

# Initialize the classifier model


# Train the classifier
for epoch in range(num_epochs):
    classifier_model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for audio_wave, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
        optimizer.zero_grad()  # Zero the parameter gradients

        # Forward pass
        features = model(audio_wave)
        outputs = classifier_model(features)

        # Compute the loss
        loss = criterion(outputs, labels)
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize

        running_loss += loss.item() * audio_wave.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    training_accuracy = correct_predictions / total_samples
    train_losses.append(running_loss / total_samples)

    # Validation phase
    classifier_model.eval()  # Set the model to evaluation mode
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for audio_wave, labels in val_loader:
            features = model(audio_wave)
            outputs = classifier_model(features)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = val_correct / val_total
    train_accuracies.append(training_accuracy)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / total_samples:.4f}, Training Accuracy: {training_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')


# Plotting the training loss and accuracy
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-', color='r')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, marker='o', linestyle='-', color='g')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

# Optionally, save the trained model
torch.save(classifier_model.state_dict(), '/home/vm-user/Downloads/Music /Music /classifier_model.pth')
# Evaluate the classifier on the validation dataset
label_to_int = {'real': 0, 'fake': 1, 'fake_clonyAI': 2, 'fake_voiceAI': 3, 'fake_voicechanger': 4, 'fake_voice_converter': 5}

def evaluate_classifier(classifier_model, val_loader, model):
    classifier_model.eval()  # Set the classifier model to evaluation mode
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for audio_wave, labels in val_loader:
            features = model(audio_wave)
            outputs = classifier_model(features)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    # Assuming labels are already integers
    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_true_labels, all_predictions, average='macro')
    f1 = f1_score(all_true_labels, all_predictions, average='macro')
    confusion = confusion_matrix(all_true_labels, all_predictions)

    # Calculate unique labels from the true labels and 44predictions
    unique_labels = sorted(np.unique(all_true_labels + all_predictions))

    return accuracy, precision, recall, f1, confusion, unique_labels

# Call evaluate_classifier
accuracy, precision, recall, f1, confusion, unique_labels = evaluate_classifier(classifier_model, val_loader, model)

# Generate display labels for the confusion matrix based on unique labels present
display_labels = [f'Class {label}' for label in unique_labels]

plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=display_labels)
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Visualize the evaluation metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(10, 5))
plt.bar(metrics, values, color=['blue', 'orange', 'green', 'red'])
plt.ylim(0, 1)
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')
plt.title('Evaluation Metrics')
plt.show()

# Print evaluation metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')
print('Confusion Matrix:')
print(confusion)
