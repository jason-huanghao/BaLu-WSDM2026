import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import expit # Sigmoid function

# --- Parameters ---
N_USERS = 5000  # Number of users
TOPICS = ['tech', 'fashion', 'sports', 'entertainment']
DIMS_PER_TOPIC = 5
TOTAL_DIMS = len(TOPICS) * DIMS_PER_TOPIC
SIMILARITY_THRESHOLDS = {
    'tech': 0.97,
    'fashion': 0.95,
    'sports': 0.95,
    'entertainment': 0.9
}
INTERFERENCE_STRENGTH = { # Corresponds loosely to e^v in the paper's Agg function
    'tech': 0.01,
    'fashion': 0.35,
    'sports': 0.2,
    'entertainment': 0.1
}

# Coefficients for simulation models (based on paper's description)
np.random.seed(421) # for reproducibility

# Treatment weights from U(-1, 1) - NOTE: No bias term in x_i^T w_t
w_t = np.random.uniform(-1, 1, TOTAL_DIMS)

# Outcome weights from N(0, 1) - NOTE: No bias term in w^T x
w_0 = np.random.randn(TOTAL_DIMS) # For baseline outcome f0
w_1 = np.random.randn(TOTAL_DIMS) # For ITE ft

# Noise levels
TREATMENT_NOISE_SD = 0.1 # Noise added *inside* sigmoid
OUTCOME_NOISE_SD = 0.5 # Final noise epsilon_i ~ N(0, 1) as per paper

def generate_features(n_users, topics, dims_per_topic):
    """Generates user features from a standard normal distribution."""
    total_dims = len(topics) * dims_per_topic
    X = np.random.randn(n_users, total_dims).astype(np.float32) + 1.0
    print(f"Generated features X with shape: {X.shape}")
    return X

def calculate_adjacencies(X, topics, dims_per_topic, thresholds):
    """Calculates adjacency matrices based on topic-specific cosine similarity."""
    n_users = X.shape[0]
    num_topics = len(topics)
    A = np.zeros((num_topics, n_users, n_users), dtype=np.float32)
    
    for i, topic in enumerate(topics):
        start_col = i * dims_per_topic
        end_col = (i + 1) * dims_per_topic
        topic_features = X[:, start_col:end_col]

        # Calculate cosine similarity for the topic
        sim_matrix = cosine_similarity(topic_features)
        adj_matrix = (sim_matrix >= thresholds[topic]).astype(np.float32)
        # Remove self-loops
        np.fill_diagonal(adj_matrix, 0)
        A[i, :, :] = adj_matrix
        print(f"  - Topic '{topic}': Threshold={thresholds[topic]}, Num edges={int(np.sum(adj_matrix)/2)}")
    print(f"Generated adjacency matrices A with shape: {A.shape}")
    return A

def assign_treatment(X, w_t, noise_sd):
    """Assigns treatment based on features: t ~ Ber(sigmoid(X @ w_t + noise))."""
    n_users = X.shape[0]
    logit = X @ w_t + np.random.normal(0, noise_sd, n_users)
    propensity_score = expit(logit) 
    t_vec = np.random.binomial(1, propensity_score).astype(np.float32)
    print(f"Assigned treatment t_vec with shape: {t_vec.shape}. Proportion treated: {np.mean(t_vec):.2f}")
    return t_vec

def calculate_indv_potential_outcomes(X, w_0, w_1):
    """
    Calculates potential outcomes y0 and y1 based on features (linear model),
    representing outcomes *before* interference and final noise.
    y0 = w_0^T x
    y1 = w_0^T x + w_1^T x = (w_0 + w_1)^T x
    """
    y0_noiseless = X @ w_0
    y1_noiseless = X @ (w_0 + w_1) # Or calculate as y0 + X @ w1

    print(f"Calculated noiseless potential outcomes y0, y1 (shapes: {y0_noiseless.shape}, {y1_noiseless.shape})")
    print(f"  - Mean y0_noiseless: {np.mean(y0_noiseless):.2f}")
    print(f"  - Mean y1_noiseless: {np.mean(y1_noiseless):.2f}")
    print(f"  - Mean Individual Treatment Effect (y1-y0)_noiseless: {np.mean(y1_noiseless - y0_noiseless):.2f}")
    return y0_noiseless, y1_noiseless

def calculate_interference_effect(A, t_vec, strengths, topics):
    """
    Calculates the total interference effect (f_s) for each user.
    This implements a simplified version of the paper's spillover,
    using weighted mean treatment of 1-hop neighbors per topic.
    """
    n_users = t_vec.shape[0]
    total_interference = np.zeros(n_users, dtype=np.float32) 

    for k, topic in enumerate(topics):
        adj_k = A[k, :, :] # Adjacency matrix for topic k (N x N)
        strength_k = strengths[topic] 
        sum_neighbor_treatments = adj_k @ t_vec # Shape (N,)
        num_neighbors = np.sum(adj_k, axis=1) # Shape (N,)

        # Calculate mean treatment of neighbors (handle division by zero for isolates)
        mean_neighbor_treatment = np.zeros_like(sum_neighbor_treatments)
        non_isolated_mask = num_neighbors > 0
        mean_neighbor_treatment[non_isolated_mask] = sum_neighbor_treatments[non_isolated_mask] / num_neighbors[non_isolated_mask]

        # Add weighted interference contribution for this topic
        total_interference += strength_k * mean_neighbor_treatment
        print(f"  - Topic '{topic}': Strength={strength_k}, Mean contribution={np.mean(strength_k * mean_neighbor_treatment):.3f}")

    print(f"Calculated total interference effect f_s. Mean: {np.mean(total_interference):.2f}")
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

    # Calculate factual outcome = baseline + ITE_if_treated + interference + noise
    y_f = np.where(t_vec, y_1, y_0)

    # Optional: Ensure outcome is non-negative if it represents something like time/count
    # y_f = np.maximum(0, y_f)

    print(f"Calculated factual outcomes y_f with shape: {y_f.shape}. Mean: {np.mean(y_f):.2f}")
    return y_f.astype(np.float32), y_0.astype(np.float32), y_1.astype(np.float32)

# --- Main Script ---
if __name__ == "__main__":
    print("--- Starting Synthetic Dataset Generation (Paper-Inspired) ---")

    # 1. Generate Features
    X = generate_features(N_USERS, TOPICS, DIMS_PER_TOPIC)

    # 2. Calculate Adjacency Matrices
    A = calculate_adjacencies(X, TOPICS, DIMS_PER_TOPIC, SIMILARITY_THRESHOLDS)

    # 3. Assign Treatment (Paper's logic with noise inside sigmoid)
    t_vec = assign_treatment(X, w_t, TREATMENT_NOISE_SD)
    # t_vec = assign_treatment(X, w_t, 0.0)

    # 4. Calculate Potential Outcomes (Noiseless, before interference)
    y0_spe, y1_spe = calculate_indv_potential_outcomes(X, w_0, w_1)

    # y_f = np.where(t_vec, y1_spe, y0_spe)
    # interference = [0]
    
    # # 5. Calculate Interference Effect (Simplified f_s)
    interference = calculate_interference_effect(A, t_vec, INTERFERENCE_STRENGTH, TOPICS)

    # # 6. Calculate Outcomes factual, y0, y1
    y_f, y0_spe, y1_spe = calculate_outcomes(y0_spe, y1_spe, t_vec, interference, OUTCOME_NOISE_SD)

    # 7. Save the data
    print("\n--- Saving Data ---")
    try:
        np.save('Syn_adjs.npy', A)
        np.save('Syn_t.npy', t_vec)
        np.save('Syn_x.npy', X.astype(np.float32)) # Save X as float32
        np.save('Syn_yf.npy', y_f)
        # Save the potential outcomes *before* interference and final noise
        np.save('Syn_y0_spe.npy', y0_spe.astype(np.float32))
        np.save('Syn_y1_spe.npy', y1_spe.astype(np.float32))
        print("Data saved successfully:")
        print("  - Syn_adjs.npy (Shape: {})".format(A.shape))
        print("  - Syn_t.npy (Shape: {})".format(t_vec.shape))
        print("  - Syn_x.npy (Shape: {})".format(X.shape))
        print("  - Syn_yf.npy (Shape: {})".format(y_f.shape))
        print("  - Syn_y0_spe.npy (Shape: {})".format(y0_spe.shape))
        print("  - Syn_y1_spe.npy (Shape: {})".format(y1_spe.shape))
    except Exception as e:
        print(f"Error saving data: {e}")

    print("\n--- Dataset Generation Complete ---")

    # Optional: Simple check of average treatment effect
    avg_outcome_treated = np.mean(y_f[t_vec == 1])
    avg_outcome_control = np.mean(y_f[t_vec == 0])
    naive_ate = avg_outcome_treated - avg_outcome_control
    # Average Individual TE (without interference or final noise)
    true_avg_ite_noiseless = np.mean(y1_spe - y0_spe)
    avg_interference_effect = np.mean(interference)

    print(f"\n--- Sanity Checks (Informational) ---")
    print(f"Naive Average Treatment Effect (Observed y_f): {naive_ate:.3f}")
    print(f"True Average Individual Treatment Effect (Noiseless y1-y0): {true_avg_ite_noiseless:.3f}")
    print(f"Average Interference Effect (f_s): {avg_interference_effect:.3f}")