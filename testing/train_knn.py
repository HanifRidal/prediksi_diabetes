import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Sample training data
X = [
    [5.1, 3.5, 1.4, 0.2],  # Class 0
    [4.9, 3.0, 1.4, 0.2],  # Class 0
    [6.2, 3.4, 5.4, 2.3],  # Class 1
    [5.9, 3.0, 5.1, 1.8]   # Class 1
]
y = [0, 0, 1, 1]  # Labels: 0 for Class 0, 1 for Class 1

# Train KNeighborsClassifier model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Save the trained model as a pickle file
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

print("Model saved to 'knn_model.pkl'.")
