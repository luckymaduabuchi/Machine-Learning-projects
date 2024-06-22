import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import os
import time
from keras.preprocessing.image import ImageDataGenerator

def resize_image(image, target_size=(128, 128)):
    return cv2.resize(image, target_size)

def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray, orientations=12, pixels_per_cell=(16, 16),
                   cells_per_block=(3, 3), block_norm='L1', transform_sqrt=True,
                   feature_vector=True)
    return features

def extract_lbp_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_combined_features(image):
    hog_features = extract_hog_features(image)
    lbp_features = extract_lbp_features(image)
    return np.concatenate((hog_features, lbp_features))

def load_images_and_labels(folder, feature_extractor, augment=False):
    images = []
    labels = []
    data_gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,
                                  height_shift_range=0.2, shear_range=0.15,
                                  zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
    for subdir in os.listdir(folder):
        label = 1 if subdir == "mask" else 0
        subdir_path = os.path.join(folder, subdir)
        for filename in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, filename)
            img = cv2.imread(img_path)
            if img is not None and len(img.shape) == 3:
                resized_img = resize_image(img)
                if augment:
                    it = data_gen.flow(np.expand_dims(resized_img, 0), batch_size=1)
                    for _ in range(5):  # Generate 5 augmented images
                        aug_img = it.next()[0].astype('uint8')
                        features = feature_extractor(aug_img)
                        images.append(features)
                        labels.append(label)
                else:
                    features = feature_extractor(resized_img)
                    images.append(features)
                    labels.append(label)
            else:
                print(f"Skipped loading a corrupted or non-image file: {filename}")
    return np.array(images), np.array(labels)

def apply_pca(X_train, X_val, n_components=0.95):
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_val_pca = pca.transform(X_val)
    return X_train_pca, X_val_pca

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

def webcam_detect(feature_extractor, knn_model):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = resize_image(frame, (300, 300))
        result = detect_mask(resized_frame, feature_extractor, knn_model)
        cv2.putText(frame, result, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def detect_mask(frame, feature_extractor, knn_model):
    features = feature_extractor(frame)
    prediction = knn_model.predict([features])
    return "Mask" if prediction[0] == 1 else "No Mask"

train_folder = '/home/vm-user/Pictures/data_resized - Copy/training'
validation_folder = '/home/vm-user/Pictures/data_resized - Copy/validation'


X_train_combined, y_train_combined = load_images_and_labels(train_folder, extract_combined_features, augment=True)
X_val_combined, y_val_combined = load_images_and_labels(validation_folder, extract_combined_features)

X_train_pca, X_val_pca = apply_pca(X_train_combined, X_val_combined)

knn_classifier_combined = KNeighborsClassifier()
param_grid = {'n_neighbors': range(1, 30)}
grid_search_combined = GridSearchCV(knn_classifier_combined, param_grid, cv=5)
grid_search_combined.fit(X_train_pca, y_train_combined)
y_pred_combined = grid_search_combined.predict(X_val_pca)

print("Validation Accuracy (Combined):", accuracy_score(y_val_combined, y_pred_combined))
print("Classification Report (Combined):\n", classification_report(y_val_combined, y_pred_combined))

plot_accuracy_vs_k(grid_search_combined, "Combined")

webcam_detect(extract_combined_features, grid_search_combined)
