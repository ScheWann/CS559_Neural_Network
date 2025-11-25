import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from scipy.stats import mode


def load_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


# Competitive Learning
class CompetitiveLearning:
    def __init__(self, n_clusters=3, learning_rate=0.1, epochs=20):
        self.n_clusters = n_clusters
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None

    def fit(self, X):
        n_samples, n_features = X.shape
        
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.weights = X[random_indices].copy()
        
        current_lr = self.lr

        for epoch in range(self.epochs):
            current_lr = self.lr * (1 - epoch / self.epochs)
            indices = np.random.permutation(n_samples)
            
            for i in indices:
                sample = X[i]

                distances = np.linalg.norm(self.weights - sample, axis=1)
                winner_idx = np.argmin(distances)
                self.weights[winner_idx] += current_lr * (sample - self.weights[winner_idx])

    def predict(self, X):
        labels = []
        for sample in X:
            distances = np.linalg.norm(self.weights - sample, axis=1)
            labels.append(np.argmin(distances))
        return np.array(labels)


# Evaluation Logic
def evaluate_clustering(cluster_labels, true_labels, algo_name="Clustering"):
    clusters = np.unique(cluster_labels)
    predicted_labels = np.zeros_like(cluster_labels)
    mapping = {}

    print(f"\n--- {algo_name} Mapping Analysis ---")
    
    for c in clusters:
        mask = (cluster_labels == c)
        if np.sum(mask) > 0:
            true_labels_in_cluster = true_labels[mask]
            majority_label = mode(true_labels_in_cluster, keepdims=True)[0][0]
            mapping[c] = majority_label
            predicted_labels[mask] = majority_label
        else:
            mapping[c] = -1

    acc = accuracy_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)
    
    print(f"Accuracy: {acc:.4f}")
    
    return predicted_labels, acc, cm


# Plot
def plot_results(X, y_true, y_cl, y_km, cm_cl, cm_km):

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    
    # Ground Truth
    axes1[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', edgecolor='k')
    axes1[0].set_title("Ground Truth Labels")
    axes1[0].set_xlabel("PC1")
    axes1[0].set_ylabel("PC2")
    
    # Competitive Learning
    axes1[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_cl, cmap='viridis', edgecolor='k')
    axes1[1].set_title("Competitive Learning Predictions")
    axes1[1].set_xlabel("PC1")
    axes1[1].set_ylabel("PC2")
    
    # K-Means
    axes1[2].scatter(X_pca[:, 0], X_pca[:, 1], c=y_km, cmap='viridis', edgecolor='k')
    axes1[2].set_title("K-Means Predictions")
    axes1[2].set_xlabel("PC1")
    axes1[2].set_ylabel("PC2")
    
    plt.tight_layout()
    plt.show()
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    
    # CL Confusion Matrix
    disp_cl = ConfusionMatrixDisplay(confusion_matrix=cm_cl, display_labels=[0, 1, 2])
    disp_cl.plot(ax=axes2[0], cmap='Blues', colorbar=False)
    axes2[0].set_title("Competitive Learning Confusion Matrix")
    
    # KMeans Confusion Matrix
    disp_km = ConfusionMatrixDisplay(confusion_matrix=cm_km, display_labels=[0, 1, 2])
    disp_km.plot(ax=axes2[1], cmap='Greens', colorbar=False)
    axes2[1].set_title("K-Means Confusion Matrix")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    X, y_true = load_data()

    cl = CompetitiveLearning(n_clusters=3, learning_rate=0.1, epochs=50)
    cl.fit(X)
    cl_clusters = cl.predict(X)
    cl_predicted_labels, cl_acc, cl_cm = evaluate_clustering(cl_clusters, y_true, "Competitive Learning")
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)
    km_clusters = kmeans.labels_
    km_predicted_labels, km_acc, km_cm = evaluate_clustering(km_clusters, y_true, "K-Means")

    plot_results(X, y_true, cl_predicted_labels, km_predicted_labels, cl_cm, km_cm)