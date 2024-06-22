import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import time

# Step 1: Load and preprocess data
def load_images_and_labels(base_dir, feature_type='HOG'):
    labels = []
    features = []
    
    for label_type in ['mask', 'unmask']:
        dir_path = os.path.join(base_dir, label_type)
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            
            if feature_type == 'HOG':
                feature, _ = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                 cells_per_block=(1, 1), visualize=True, channel_axis=-1)
            else:
                continue
            
            features.append(feature)
            labels.append(1 if label_type == 'mask' else 0)
    
    return np.array(features), np.array(labels)

# Step 2: Scale features
def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled

# Step 3: Perform grid search
def perform_grid_search(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.001, 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf']
    }
    grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5, return_train_score=True)
    start_time = time.time()  # Start timing
    grid_search.fit(X_train, y_train)
    end_time = time.time()  # End timing
    training_time = end_time - start_time
    print("Training completed in {:.2f} seconds".format(training_time))
    print("Best parameters found: ", grid_search.best_params_)
    return grid_search.best_estimator_, training_time, grid_search.cv_results_

# Step 4: Evaluate the model
def evaluate_model(model, X_train, y_train, X_test, y_test, cv_results):
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print("Training Accuracy: {:.2f}%".format(train_accuracy * 100))
    print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
    print("Classification Report (Test Set):")
    print(classification_report(y_test, test_predictions))
    mean_train_loss = -np.mean(cv_results['mean_train_score'])
    print("Mean Cross-Validated Training Loss: {:.4f}".format(mean_train_loss))

# Step 5: Implement webcam integration
def use_webcam(model):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        feature, _ = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                         cells_per_block=(1, 1), visualize=True, channel_axis=-1)
        
        prediction = model.predict([feature])
        label = 'Mask' if prediction == 1 else 'Unmask'
        
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Webcam', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function to tie all steps
def main():
    training_features, training_labels = load_images_and_labels('/home/vm-user/Pictures/data_resized - Copy/training')
    validation_features, validation_labels = load_images_and_labels('/home/vm-user/Pictures/data_resized - Copy/validation')
 

    
    # Split validation into actual validation and test sets
    val_features, test_features, val_labels, test_labels = train_test_split(
        validation_features, validation_labels, test_size=0.5, random_state=42
    )
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(training_features, val_features, test_features)
    
    # Perform grid search
    best_model, training_time, cv_results = perform_grid_search(X_train_scaled, training_labels)
    
    # Evaluate the model
    evaluate_model(best_model, X_train_scaled, training_labels, X_test_scaled, test_labels, cv_results)
    
    # Use webcam
    use_webcam(best_model)

if __name__ == "__main__":
    main()
