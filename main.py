import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 1: Generate synthetic LiDAR-like data with balanced traffic labels
def generate_synthetic_lidar_data(num_points):
    # Randomly generate vehicle counts (1 to 100)
    vehicle_count = np.random.randint(1, 100, num_points)
    # Randomly generate traffic densities (0 to 10)
    traffic_density = np.random.uniform(0, 10, num_points)

    # Create a more balanced set of traffic labels
    traffic_label = []
    for count in vehicle_count:
        if count < 30:
            traffic_label.append('light')
        elif count < 70:
            traffic_label.append('moderate')
        else:
            traffic_label.append('heavy')

    return vehicle_count, traffic_density, traffic_label

# Step 2: Generate synthetic data
num_points = 100  # Specify the number of synthetic samples
vehicle_count, traffic_density, traffic_label = generate_synthetic_lidar_data(num_points)

# Step 3: Prepare data for SVM model
data = {
    'vehicle_count': vehicle_count,
    'traffic_density': traffic_density,
    'traffic_label': traffic_label
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert traffic labels to numerical values
df['traffic_label'] = df['traffic_label'].map({'light': 0, 'moderate': 1, 'heavy': 2})

# Step 4: Define features and target
X = df[['vehicle_count', 'traffic_density']]
y = df['traffic_label']

# Step 5: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Train SVM Model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Step 7: Evaluate Model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 8: Adjust traffic signal function
def adjust_traffic_signal(traffic_class):
    if traffic_class == 0:
        print("Traffic is light: Green light for 30 seconds.")
    elif traffic_class == 1:
        print("Traffic is moderate: Green light for 45 seconds.")
    else:
        print("Traffic is heavy: Green light for 60 seconds.")

# Step 9: Simulate multiple traffic predictions
num_simulations = 5  # Specify how many simulations you want

for _ in range(num_simulations):
    # Generate random vehicle count and density for new sensor data
    random_vehicle_count = np.random.randint(1, 100)
    random_traffic_density = np.random.uniform(0, 10)
    new_sensor_data = pd.DataFrame([[random_vehicle_count, random_traffic_density]], columns=['vehicle_count', 'traffic_density'])

    # Predict traffic class based on the new data
    predicted_traffic_class = svm_model.predict(new_sensor_data)[0]

    # Print predicted traffic class
    print(f"Predicted Traffic Class: {predicted_traffic_class} (Vehicle Count: {random_vehicle_count}, Traffic Density: {random_traffic_density:.2f})")

    # Adjust the traffic signal based on the prediction
    adjust_traffic_signal(predicted_traffic_class)
