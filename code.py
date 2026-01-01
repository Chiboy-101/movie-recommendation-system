# Load necessary modules
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD

# Load the dataset
movies = pd.read_csv("movies.csv")

# Check for missing values
movies.isnull().sum()

# Convert the movie genres to numerical features
tfidf = TfidfVectorizer(stop_words="english")
movies_tfidf = tfidf.fit_transform(movies["genres"])

# Elbow Method
wcss = []
k_range = range(1, 13)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=28)
    kmeans.fit(movies_tfidf)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()


# Silhouette Score
for k in range(2, 13):
    km = KMeans(n_clusters=k, random_state=28)
    labels = km.fit_predict(movies_tfidf)
    score = silhouette_score(movies_tfidf, labels)
    print(f"k={k}, silhouette score={score:.4f}")


# Final KMeans Clustering
k = 10  # chosen ideal number of clusters
kmeans = KMeans(n_clusters=10, random_state=28)
movies["cluster"] = kmeans.fit_predict(movies_tfidf)

movies.head()


# A movie recommendation Function
def recommend_movie(movie_title, top_n=10):
    idx = movies[movies["title"].str.contains(movie_title, case=False)].index[0]

    movie_cluster = movies.loc[idx, "cluster"]

    cluster_indices = movies[movies["cluster"] == movie_cluster].index

    similarity = cosine_similarity(
        movies_tfidf[idx],  # type: ignore
        movies_tfidf[cluster_indices],  # type: ignore
    ).flatten()

    similar_indices = cluster_indices[similarity.argsort()[-top_n - 1 : -1][::-1]]

    return movies.loc[similar_indices, ["title", "genres"]]


recommend_movie("Toy Story")  # Testing the function on a movie tile


# Dimensionality Reduction for Visualization
svd = TruncatedSVD(n_components=2, random_state=28)
tfidf_2d = svd.fit_transform(movies_tfidf)

# Plot
plt.figure(figsize=(15, 10))

plt.scatter(
    tfidf_2d[:, 0], tfidf_2d[:, 1], c=movies["cluster"], cmap="tab10", s=50, alpha=0.6
)

# Project cluster centers correctly
centers_2d = svd.transform(kmeans.cluster_centers_)
plt.scatter(
    centers_2d[:, 0], centers_2d[:, 1], c="black", s=200, marker="X", label="Centers"
)

plt.title("Movie Clusters with Cluster Centers (2D SVD Projection)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar(label="Cluster")
plt.legend()
plt.show()
