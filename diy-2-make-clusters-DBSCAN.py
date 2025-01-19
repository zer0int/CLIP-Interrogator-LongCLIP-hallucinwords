import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import clip
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# Input words (or sentences) must each be <77 tokens and separated by a newline
input_words_file = "words_processing/raw_words.txt"
output_folder = "words_processing/clusters"
plot_folder = "words_processing/plots"
embedding_file = "embeds/all.pt"
leftover_words_file = "words_processing/clusters/leftover_words.txt"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)
os.makedirs("embeds", exist_ok=True)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

# Load input words (prompts)
with open(input_words_file, "r", encoding="utf-8") as f:
    all_words = [line.strip() for line in f if line.strip()]

# Generate / load embeddings
if os.path.exists(embedding_file):
    print("Loading precomputed embeddings...")
    embeddings = torch.load(embedding_file)
else:
    print("Computing embeddings...")
    embeddings = []
    for word in all_words:
        text = clip.tokenize(word).to(device)
        with torch.no_grad():
            embedding = model.encode_text(text).cpu().numpy()
        embeddings.append(embedding[0])
    embeddings = np.array(embeddings)
    torch.save(embeddings, embedding_file)
    print(f"Embeddings saved to {embedding_file}")

# Dimensionality reduction
print("Reducing dimensionality...")
pca = PCA(n_components=50)  # Adjust as needed; typically 20-50
reduced_embeddings = pca.fit_transform(embeddings)

#Normalize embeddings
scaler = StandardScaler()
reduced_embeddings = scaler.fit_transform(reduced_embeddings)

# Remove features with low variance, if needed.
#selector = VarianceThreshold(threshold=0.01)
#selected_embeddings = selector.fit_transform(reduced_embeddings)

# Visualize all embeddings
plt.figure(figsize=(10, 10))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=1, alpha=0.3)
plt.title("Overview of Reduced Embeddings")
plt.savefig(os.path.join(plot_folder, "all-overview.png"))
plt.close()

# Iterative DBSCAN clustering
print("Starting iterative clustering...")
remaining_indices = np.arange(len(all_words))
iteration = 0

while len(remaining_indices) > 0:
    iteration += 1
    print(f"Iteration {iteration}: Clustering {len(remaining_indices)} words...")

    # Select remaining embeddings and words
    current_embeddings = reduced_embeddings[remaining_indices]
    current_words = [all_words[i] for i in remaining_indices]

    # Perform DBSCAN clustering
    eps = 0.14 + 0.01 * (iteration - 1)  # Relax eps parameter; try [0.04-0.20] + [0.01-0.04]
    print(f"eps:{eps}")
    min_samples = 50  # Minimum number of samples for a cluster; try 3-50 [default: 5]
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", algorithm="brute")
    labels = dbscan.fit_predict(current_embeddings)

    # Group words into clusters
    clusters = {}
    noise_words = []
    for idx, label in enumerate(labels):
        if label == -1:
            noise_words.append(current_words[idx])
        else:
            clusters.setdefault(label, []).append(current_words[idx])

    # Save clusters
    for cluster_id, cluster_words in clusters.items():
        cluster_file = os.path.join(output_folder, f"cluster_iter_{iteration}_id_{cluster_id}.txt")
        with open(cluster_file, "w", encoding="utf-8") as f:
            f.write("\n".join(cluster_words))

        # Visualize each cluster
        cluster_indices = [remaining_indices[idx] for idx, label in enumerate(labels) if label == cluster_id]
        cluster_points = reduced_embeddings[cluster_indices]
        plt.figure(figsize=(10, 10))
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=5, alpha=0.5)

        # Highlight 10 random points in the cluster, with text labels
        if len(cluster_indices) > 10:
            highlight_indices = np.random.choice(cluster_indices, 10, replace=False)
        else:
            highlight_indices = cluster_indices

        for idx in highlight_indices:
            plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], color="red")
            plt.text(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], all_words[idx], fontsize=8)

        plt.title(f"Cluster {cluster_id} (Iteration {iteration})")
        plt.savefig(os.path.join(plot_folder, f"cluster_iter_{iteration}_id_{cluster_id}.png"))
        plt.close()

    # Check stopping condition: If leftover words are <= 0.01 (i.e., 1%) of original words
    if len(noise_words) <= 0.01 * len(all_words):
        with open(leftover_words_file, "w", encoding="utf-8") as f:
            f.write("\n".join(noise_words))
        print(f"Clustering complete. Leftover words saved to {leftover_words_file}")
        break

    # Update remaining indices for the next iteration
    remaining_indices = [remaining_indices[idx] for idx, label in enumerate(labels) if label == -1]

    print(f"Iteration {iteration} completed. {len(remaining_indices)} words remain unclustered.")

print(f"Clustering complete. Check {output_folder} & {plot_folder} for results.")