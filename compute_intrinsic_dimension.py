import numpy as np
from sklearn.neighbors import NearestNeighbors

def estimate_intrinsic_dimension(data, k=5):
    """
    Estimate the intrinsic dimension of a dataset using the Maximum Likelihood Estimation (MLE) method.

    Parameters:
    - data: numpy array of shape [n_samples, n_features]
        The input data, where n_samples is the number of samples and n_features is the dimensionality of each sample.
    - k: int, optional (default=5)
        The number of nearest neighbors to consider for the estimation.

    Returns:
    - m: float
        The estimated intrinsic dimension of the dataset.
    """
    n_samples = data.shape[0]

    # Fit Nearest Neighbors model to the data
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(data)

    # Compute the distances and indices of the k nearest neighbors
    distances, indices = nbrs.kneighbors(data)  # distances shape: [n_samples, k+1]

    # Exclude the first column (distance to itself, which is zero)
    distances = distances[:, 1:]  # shape: [n_samples, k]

    # Take the k-th nearest distance for each sample
    r_k = distances[:, -1]  # shape: [n_samples]

    # Initialize a list to store m_i for each sample
    m_i = []

    # Compute m_i for each sample
    for i in range(n_samples):
        r_ki = r_k[i]
        sum_log = 0.0
        for j in range(k):
            sum_log += np.log(r_ki / distances[i, j])
        m_i.append(sum_log / (k - 1))

    m_i = np.array(m_i)

    # Compute the global intrinsic dimension estimate m
    m = n_samples / np.sum(m_i)

    return m

if __name__ == "__main__":
    # Example usage:

    n_samples = 10
    n_features = 8 
    latent_embeddings = np.random.rand(n_samples, n_features)

    k = 6 
    intrinsic_dim = estimate_intrinsic_dimension(latent_embeddings, k=k)

    print("Estimated intrinsic dimension:", intrinsic_dim)
