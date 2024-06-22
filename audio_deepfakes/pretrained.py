import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import os
import time

def resize_image(image, target_size=(128, 128)):
    return cv2.resize(image, target_size)

# Function to extract HOG features from an image
def extract_hog_features(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate HOG features
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True,
                   feature_vector=True)
    return features

# Function to extract LBP features from an image
def extract_lbp_features(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate LBP features
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# Function to load images and extract features
def load_images_and_labels(folder, feature_extractor):
    images = []
    labels = []
    for subdir in os.listdir(folder):
        label = 1 if subdir == "mask" else 0
        subdir_path = os.path.join(folder, subdir)
        for filename in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, filename)
            img = cv2.imread(img_path)
            if img is not None and len(img.shape) == 3:  # Ensure the image is loaded and is in color
                resized_img = resize_image(img)
                features = feature_extractor(resized_img)
                images.append(features)
                labels.append(label)
            else:
                print(f"Skipped loading a corrupted or non-image file: {filename}")
    return np.array(images), np.array(labels)


# Load training and validation data
train_folder = '/home/vm-user/Pictures/data_resized - Copy/training'
validation_folder = '/home/vm-user/Pictures/data_resized - Copy/validation'

# Feature extraction using HOG
X_train_hog, y_train_hog = load_images_and_labels(train_folder, extract_hog_features)
X_val_hog, y_val_hog = load_images_and_labels(validation_folder, extract_hog_features)

# Feature extraction using LBP
X_train_lbp, y_train_lbp = load_images_and_labels(train_folder, extract_lbp_features)
X_val_lbp, y_val_lbp = load_images_and_labels(validation_folder, extract_lbp_features)

# Initialize KNN classifiers
knn_classifier_hog = KNeighborsClassifier()
knn_classifier_lbp = KNeighborsClassifier()

# Start the timer for HOG
start_time_hog = time.time()
# Hyperparameter tuning using GridSearchCV for HOG features
param_grid = {'n_neighbors': range(1, 30)}
grid_search_hog = GridSearchCV(knn_classifier_hog, param_grid, cv=5)
grid_search_hog.fit(X_train_hog, y_train_hog)
end_time_hog = time.time()

# Calculate the training time for HOG
training_time_hog = end_time_hog - start_time_hog
print("Training Time (HOG):", training_time_hog, "seconds")

# Start the timer for LBP
start_time_lbp = time.time()
# Hyperparameter tuning using GridSearchCV for LBP features
grid_search_lbp = GridSearchCV(knn_classifier_lbp, param_grid, cv=5)
grid_search_lbp.fit(X_train_lbp, y_train_lbp)
end_time_lbp = time.time()

# Calculate the training time for LBP
training_time_lbp = end_time_lbp - start_time_lbp
print("Training Time (LBP):", training_time_lbp, "seconds")

# Evaluate on validation set for HOG features
y_pred_hog = grid_search_hog.predict(X_val_hog)
accuracy_hog = accuracy_score(y_val_hog, y_pred_hog)
print("Validation Accuracy (HOG):", accuracy_hog)
print("Classification Report (HOG):\n", classification_report(y_val_hog, y_pred_hog))

# Evaluate on validation set for LBP features
y_pred_lbp = grid_search_lbp.predict(X_val_lbp)
accuracy_lbp = accuracy_score(y_val_lbp, y_pred_lbp)
print("Validation Accuracy (LBP):", accuracy_lbp)
print("Classification Report (LBP):\n", classification_report(y_val_lbp, y_pred_lbp))

# Plotting accuracy for different k values
def plot_accuracy_vs_k(grid_search_result, feature_type):
    k_values = range(1, 30)
    accuracies = grid_search_result.cv_results_['mean_test_score']
    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='blue')
    plt.title('KNN Accuracy vs. k for ' + feature_type)
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

plot_accuracy_vs_k(grid_search_hog, "HOG")
plot_accuracy_vs_k(grid_search_lbp, "LBP")

# Function to detect mask using the trained KNN model
def detect_mask(frame, feature_extractor, knn_model):
    features = feature_extractor(frame)
    prediction = knn_model.predict([features])
    return "Mask" if prediction[0] == 1 else "No Mask"

# Webcam detection
def webcam_detect(feature_extractor, knn_model):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to a smaller size for faster processing
        resized_frame = cv2.resize(frame, (300, 300))
        
        # Detect mask
        result = detect_mask(resized_frame, feature_extractor, knn_model)
        
        # Display result on the frame
        cv2.putText(frame, result, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start webcam detection with HOG features
webcam_detect(extract_hog_features, grid_search_hog)

# Start webcam detection with LBP features
# webcam_detect(extract_lbp_features, grid_search_lbp)