from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def sim_rel_outcomes(relation_sim, X, T, n_rel):

    thresholds = [np.random.uniform(0.95, 0.98) for _ in range(n_rel)]
    A = simulate_A_Y(X, n_rel, thresholds)    #[n_rel, N, N]
    
    # 2. Calculate Potential Outcomes (Noiseless, before interference)
    w_0 = np.random.randn(X.shape[1]) 
    w_1 = np.random.randn(X.shape[1])
    y0_spe, y1_spe = calculate_indv_potential_outcomes(X, w_0, w_1)
    
    # 3. Calculate Interference Effect (Simplified f_s)
    # strengths = [np.random.uniform(0.0, 1.0/n_rel*2) for _ in range(n_rel)]
    x = np.random.exponential(scale=2, size=n_rel)
    strengths = np.exp(x) / np.sum(np.exp(x))
    
    # strengths = [np.random.uniform(0.0, 1.0) for _ in range(n_rel)]
    # strengths = [e/sum(strengths) for e in strengths]

    interference = calculate_interference_effect(A, T, strengths)

    # 4. Calculate Outcomes factual, y0, y1
    Y, Y0, Y1 = calculate_outcomes(y0_spe, y1_spe, T, interference, noise_sd=0.5)
    return X, A, T, Y, Y1, Y0

def simulate_A_Y(X: np.ndarray, n_rel: int, thresholds: list):
    """
        np.ndarray: A 3D numpy array 'A' of shape (n_rel, n_users, n_users),
                    where A[i, :, :] is the adjacency matrix for the i-th
                    relationship type.
    """
    assert n_rel > 0 
    assert n_rel == len(thresholds)
    
    n_users = X.shape[0]
    total_features = X.shape[1]
    A = np.zeros((n_rel, n_users, n_users))

    # --- Generate col_indices_per_rel by random distribution ---
    col_indices_per_rel = [[] for _ in range(n_rel)]
    
    all_col_indices = np.arange(total_features)
    for col_idx in all_col_indices:
        random_rel_idx = np.random.randint(0, n_rel)
        col_indices_per_rel[random_rel_idx].append(col_idx)
    
    for i in range(n_rel):
        topic_features = X[:, col_indices_per_rel[i]]
        if topic_features.shape[1] == 0:
            A[i, :, :] = np.zeros((n_users, n_users))
            continue
        
        sim_matrix = cosine_similarity(topic_features)
        adj_matrix = (sim_matrix >= thresholds[i]).astype(int)
        np.fill_diagonal(adj_matrix, 0)
        A[i, :, :] = adj_matrix
    return A


def calculate_indv_potential_outcomes(X, w_0, w_1):
    """
    Calculates potential outcomes y0 and y1 based on features (linear model),
    representing outcomes *before* interference and final noise.
    y0 = w_0^T x
    y1 = w_0^T x + w_1^T x = (w_0 + w_1)^T x
    """
    y0_noiseless = X @ w_0
    # y1_noiseless = X @ (w_0 + w_1)
    y1_noiseless = X @ (w_0 + w_1) + 0.5    # Or calculate as y0 + X @ w1

    # print(f"Calculated noiseless potential outcomes y0, y1 (shapes: {y0_noiseless.shape}, {y1_noiseless.shape})")
    # print(f"  - Mean y0_noiseless: {np.mean(y0_noiseless):.2f}")
    # print(f"  - Mean y1_noiseless: {np.mean(y1_noiseless):.2f}")
    # print(f"  - Mean Individual Treatment Effect (y1-y0)_noiseless: {np.mean(y1_noiseless - y0_noiseless):.2f}")
    return y0_noiseless, y1_noiseless

def calculate_interference_effect(A, t_vec, strengths):
    """
    Calculates the total interference effect (f_s) for each user.
    This implements a simplified version of the paper's spillover,
    using weighted mean treatment of 1-hop neighbors per topic.
    """
    n_users = t_vec.shape[0]
    total_interference = np.zeros(n_users, dtype=np.float32) 

    for k, strength_k in enumerate(strengths):
        adj_k = A[k, :, :] # Adjacency matrix for topic k (N x N)
        sum_neighbor_treatments = adj_k @ t_vec # Shape (N,)
        num_neighbors = np.sum(adj_k, axis=1) # Shape (N,)

        # Calculate mean treatment of neighbors (handle division by zero for isolates)
        mean_neighbor_treatment = np.zeros_like(sum_neighbor_treatments)
        non_isolated_mask = num_neighbors > 0
        mean_neighbor_treatment[non_isolated_mask] = sum_neighbor_treatments[non_isolated_mask] / num_neighbors[non_isolated_mask]

        # Add weighted interference contribution for this topic
        total_interference += strength_k * mean_neighbor_treatment
        print(f"  - Topic '{k}': Strength={strength_k}, Mean contribution={np.mean(strength_k * mean_neighbor_treatment):.3f}")

    # print(f"Calculated total interference effect f_s. Mean: {np.mean(total_interference):.2f}")
    return total_interference.astype(np.float32) # Cast back to float32 if needed

def calculate_outcomes(y0_noiseless, y1_noiseless, t_vec, interference_effect, noise_sd):
    """
    Calculates factual outcomes based on paper's formula:
    y_f = f0 + ft + fs + epsilon
        = y0_noiseless + t * (y1_noiseless - y0_noiseless) + fs + noise
    """
    # ite_noiseless = y1_noiseless - y0_noiseless 
    y_0 = y0_noiseless + interference_effect + np.random.normal(0, noise_sd, len(t_vec))
    y_1 = y1_noiseless + interference_effect + np.random.normal(0, noise_sd, len(t_vec))

    print("ite:", np.mean(y1_noiseless - y0_noiseless), 
          "\t rte:", np.mean(interference_effect))
    print("mean Y0:", np.mean(y_0), "\t mean Y1:", np.mean(y_1), "causation:", np.mean(y_1) - np.mean(y_0))
    # Calculate factual outcome = baseline + ITE_if_treated + interference + noise
    y_f = np.where(t_vec, y_1, y_0)

    # Optional: Ensure outcome is non-negative if it represents something like time/count
    # y_f = np.maximum(0, y_f)

    # print(f"Calculated factual outcomes y_f with shape: {y_f.shape}. Mean: {np.mean(y_f):.2f}")
    print("mean Y|T=0:", np.mean([y for t, y in zip(t_vec, y_f) if t < 0.5]), 
          "\t mean Y|T=1:", np.mean([y for t, y in zip(t_vec, y_f) if t > 0.5]), 
          "Association:", np.mean([y for t, y in zip(t_vec, y_f) if t > 0.5]) - np.mean([y for t, y in zip(t_vec, y_f) if t < 0.5]))
    return y_f.astype(np.float32), y_0.astype(np.float32), y_1.astype(np.float32)
