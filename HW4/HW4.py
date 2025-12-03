import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

path=r"C:\Users\User\Downloads\胡樂麒 資料探勘HW4\kagglehub\datasets\uciml\news-aggregator-dataset\versions\1"
print("Path to dataset files:", path)
file_path = os.path.join(path, "uci-news-aggregator.csv")


df = pd.read_csv(file_path, header=None)

df.columns = [
    "ID", "TITLE", "URL", "PUBLISHER",
    "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"
]
documents = df["TITLE"].astype(str).tolist()[:500]
print("Documents loaded:", len(documents))


from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline
n_clusters = 10
pipeline = Pipeline([('feature_extraction', TfidfVectorizer(max_df=0.4)),('clusterer', KMeans(n_clusters=n_clusters))])

pipeline.fit(documents)
labels = pipeline.predict(documents)

from collections import Counter
c = Counter(labels)
for cluster_number in range(n_clusters):
 print("Cluster {} contains {} samples".format(cluster_number,c[cluster_number]))

 pipeline.named_steps['clusterer'].inertia_

inertia_scores = []
n_cluster_values = list(range(2, 20))
for n_clusters in n_cluster_values:
 cur_inertia_scores = []
 X = TfidfVectorizer(max_df=0.4).fit_transform(documents)
 for i in range(10):
    km = KMeans(n_clusters=n_clusters).fit(X)
    cur_inertia_scores.append(km.inertia_)
 inertia_scores.append(cur_inertia_scores)

n_clusters = 6
pipeline = Pipeline([('feature_extraction',
 TfidfVectorizer(max_df=0.4)),
 ('clusterer', KMeans(n_clusters=n_clusters))])
pipeline.fit(documents)
labels = pipeline.predict(documents)


terms = pipeline.named_steps['feature_extraction'].get_feature_names_out()

c = Counter(labels)

for cluster_number in range(n_clusters):
 print("Cluster {} contains {} samples".format(cluster_number,c[cluster_number]))

 print(" Most important terms")
 centroid = pipeline.named_steps['clusterer'].cluster_centers_[cluster_number]
 most_important = centroid.argsort()

 for i in range(5):
    term_index = most_important[-(i+1)]
    print(" {0}) {1} (score: {2:.4f})".format(i+1, terms[term_index], centroid[term_index]))

X = pipeline.transform(documents)

from sklearn.metrics import silhouette_score
def bcubed_precision(labels, true_labels):
    N = len(labels)
    precision = 0
    
    for i in range(N):
        same_cluster = [j for j in range(N) if labels[j] == labels[i]]
        same_class = [j for j in range(N) if true_labels[j] == true_labels[i]]
        precision += len(set(same_cluster) & set(same_class)) / len(same_cluster)

    return precision / N


def bcubed_recall(labels, true_labels):
    N = len(labels)
    recall = 0

    for i in range(N):
        same_cluster = [j for j in range(N) if labels[j] == labels[i]]
        same_class = [j for j in range(N) if true_labels[j] ==  true_labels[i]]
        recall += len(set(same_cluster) & set(same_class)) / len(same_class)

    return recall / N


true_labels = df["CATEGORY"].astype(str).tolist()[:len(labels)]
labels_kmeans = labels   
print("\n=== Number of Clusters (KMeans) ===")
print("KMeans clusters:", len(set(labels_kmeans)))
sil_kmeans = silhouette_score(X, labels_kmeans)
bc_prec_kmeans = bcubed_precision(labels_kmeans, true_labels)
bc_rec_kmeans = bcubed_recall(labels_kmeans, true_labels)
print("\n=== KMeans Evaluation ===")
print("Silhouette Score:", sil_kmeans)
print("BCubed Precision:", bc_prec_kmeans)
print("BCubed Recall:", bc_rec_kmeans)

# ========= b) KMeans 找出六大主題 ==========
from collections import Counter

cluster_counts = Counter(labels)

# 取出資料量最大的六個 clusters
top6_clusters = [c for c, _ in cluster_counts.most_common(6)]
print("\nTop 6 topic clusters:", top6_clusters)

vectorizer = pipeline.named_steps['feature_extraction']
terms = vectorizer.get_feature_names_out()
model = pipeline.named_steps['clusterer']

print("\n===== Six Main Topics =====")

for cluster_id in top6_clusters:
    centroid = model.cluster_centers_[cluster_id]
    
    # 找出cluster的 8 個關鍵詞
    top_indices = centroid.argsort()[-8:][::-1]
    keywords = [terms[i] for i in top_indices]

    print(f"\nTopic Cluster {cluster_id}:")
    print("Keywords:", ", ".join(keywords))
#----------------------------------------------------------------------


from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from collections import Counter

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components


class EAC(BaseEstimator, ClusterMixin):
    def __init__(self, 
                 n_clusterings=20,          # 要跑幾次 KMeans
                 n_clusters_range=(3, 10),  
                 final_k=6,                 
                 random_state=42):
        self.n_clusterings = n_clusterings
        self.n_clusters_range = n_clusters_range
        self.final_k = final_k
        self.random_state = random_state
        self.labels_ = None

    def fit(self, X, y=None):
        n_samples = X.shape[0]
        clusterings = []

        # ---- 1. 多次 KMeans ----
        for i in range(self.n_clusterings):
            k = np.random.randint(*self.n_clusters_range)
            km = KMeans(n_clusters=k, 
                        n_init='auto',
                        random_state=self.random_state + i)
            labels = km.fit_predict(X)
            clusterings.append(labels)

        # ---- 2. 建 co-association matrix ----
        co_mat = np.zeros((n_samples, n_samples))
        for labels in clusterings:
            for lab in np.unique(labels):
                idx = np.where(labels == lab)[0]
                co_mat[np.ix_(idx, idx)] += 1

        co_mat = co_mat / self.n_clusterings

        # ---- 3. 變距離矩陣 -> linkage ----
        dist_mat = 1 - co_mat
        condensed_dist = squareform(dist_mat, checks=False)
        Z = linkage(condensed_dist, method='average')

        # ---- 4. 切成 final_k 群 ----
        labels_final = fcluster(Z, t=self.final_k, criterion='maxclust') - 1
  
        self.labels_ = labels_final
        self.n_components = len(np.unique(labels_final))

        return self

    def predict(self, X):
        if self.labels_ is None:
            raise Exception("EAC must be fitted first")
        return self.labels_  
pipeline = Pipeline([
    ('feature_extraction', TfidfVectorizer(max_df=0.4)),
    ('clusterer', EAC(n_clusterings=20, 
                      n_clusters_range=(3, 10),
                      final_k=6))
])
pipeline.fit(documents)
labels = pipeline.named_steps['clusterer'].labels_

labels_eac = labels.copy()

bc_prec_eac = bcubed_precision(labels_eac, true_labels)
bc_rec_eac = bcubed_recall(labels_eac, true_labels)

print("\n=== Number of Clusters (EAC) ===")
print("EAC clusters:", len(set(labels_eac)))

print("\n=== EAC Evaluation ===")
if len(set(labels_eac)) > 1:
    sil_eac = silhouette_score(X, labels_eac)
    print("Silhouette:", sil_eac)
else:
    print("EAC only found 1 cluster.")
print("BCubed Precision:", bc_prec_eac)
print("BCubed Recall:", bc_rec_eac)

# ========== b) EAC 找出六大主題 ==========

print("\n===== EAC — Six Main Topics =====")


cluster_counts_eac = Counter(labels_eac)


top6_eac_clusters = [c for c, _ in cluster_counts_eac.most_common(6)]
print("Top 6 EAC clusters:", top6_eac_clusters)


vectorizer_eac = pipeline.named_steps['feature_extraction']
terms_eac = vectorizer_eac.get_feature_names_out()


X_eac = vectorizer_eac.transform(documents)

for cluster_id in top6_eac_clusters:
    mask = (labels_eac == cluster_id)
    cluster_matrix = X_eac[mask]

   
    mean_tfidf = np.asarray(cluster_matrix.mean(axis=0)).ravel()

    top_indices = mean_tfidf.argsort()[-8:][::-1]
    keywords = [terms_eac[i] for i in top_indices]

    print(f"\nEAC Cluster {cluster_id}:")
    print("Keywords:", ", ".join(keywords))
"""from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics import silhouette_score
from collections import Counter

# 假設 bcubed_precision, bcubed_recall, documents, true_labels 已經定義和載入
# 假設 max_df 變數未定義，我們將其替換為具體數值。

# -------------------------
# ① 建立共現矩陣 (保持不變)
# -------------------------
def create_coassociation_matrix(labels):
    rows, cols = [], []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        for i in idx:
            for j in idx:
                rows.append(i)
                cols.append(j)
    data = np.ones(len(rows))
    return csr_matrix((data, (rows, cols)), shape=(len(labels), len(labels)))

# -------------------------
# ② EAC 類別 (參數強化)
# -------------------------
class EAC(BaseEstimator, ClusterMixin):

    # 參數強化：n_clusterings=50, cut_threshold=0.95
    def __init__(self, n_clusterings=50, cut_threshold=0.7, n_clusters_range=(3, 10)):
        self.n_clusterings = n_clusterings
        self.cut_threshold = cut_threshold
        self.n_clusters_range = n_clusters_range
        self.labels_ = None
        self.n_components = 1 # 初始化

    def fit(self, X, y=None):
        # ----- 1. 多次 KMeans -----
        C = sum(
            create_coassociation_matrix(self._single_clustering(X))
            for _ in range(self.n_clusterings)
        )

        # ----- 2. 正規化到 [0,1] -----
        C = C / float(self.n_clusterings)

        # ----- 3. 最小生成樹 MST -----
        mst = minimum_spanning_tree(-C)

        # ----- 4. 切割 threshold (使用強化參數 0.95) -----
        # 移除相似度 C < 0.95 的邊
        mst.data[mst.data > -self.cut_threshold] = 0
        mst.eliminate_zeros()

        # ----- 5. 找 connected components → cluster labels -----
        self.n_components, self.labels_ = connected_components(mst)
        
        if self.n_components == 1:
            print(f"❌ 警告: EAC 激進剪切 (固定閾值={self.cut_threshold}) 仍然只找到 1 個群集。")

        return self

    def _single_clustering(self, X):
        k = np.random.randint(*self.n_clusters_range)
        # 使用 n_init='auto' 遵循新版 sklearn 慣例
        km = KMeans(n_clusters=k, n_init='auto', random_state=None) 
        return km.fit_predict(X)

    def predict(self, X):
        return self.labels_

# -------------------------
# ③ Pipeline 執行 (使用強化參數)
# -------------------------

pipeline = Pipeline([
    ('feature_extraction', TfidfVectorizer(max_df=0.6, stop_words='english', ngram_range=(1, 2))),

    ('clusterer', EAC(n_clusterings=50, cut_threshold=0.7, n_clusters_range=(3, 10)))
])

# 假設 documents 變數已載入
pipeline.fit(documents)
labels_eac = pipeline.named_steps['clusterer'].labels_
number_of_clusters_eac = pipeline.named_steps['clusterer'].n_components


print("\n=== Number of Clusters (EAC) ===")
print("EAC clusters:", number_of_clusters_eac)

# ============================================================
# 4. 評估 (silhouette / BCubed)
# ============================================================
vec = pipeline.named_steps['feature_extraction']
X_eac = vec.transform(documents)

if number_of_clusters_eac > 1:
    sil_eac = silhouette_score(X_eac, labels_eac)
    print("Silhouette Score:", sil_eac)
else:
    print("Silhouette Score: only 1 cluster")

# 假設 bcubed_precision/recall 和 true_labels 已定義
bc_prec_eac = bcubed_precision(labels_eac, true_labels)
bc_rec_eac = bcubed_recall(labels_eac, true_labels)

print("BCubed Precision:", bc_prec_eac)
print("BCubed Recall:", bc_rec_eac)


# ============================================================
# 5. b 小題：六大主題
# ============================================================
print("\n===== EAC — Six Main Topics =====")

# 確保只有在找到超過一個群集時才進行主題提取
if number_of_clusters_eac > 1:
    cluster_counts_eac = Counter(labels_eac)
    top6_eac_clusters = [c for c, _ in cluster_counts_eac.most_common(6)]
    print("Top 6 EAC clusters:", top6_eac_clusters)

    terms_eac = vec.get_feature_names_out()

    for cid in top6_eac_clusters:
        mask = (labels_eac == cid)
        cluster_matrix = X_eac[mask]
        
        # 處理空群集
        if cluster_matrix.shape[0] == 0:
            print(f"\nEAC Cluster {cid}: (Empty)")
            continue

        mean_tfidf = np.asarray(cluster_matrix.mean(axis=0)).ravel()
        # 確保 top_idx 不超過可用特徵數
        top_k = min(8, len(terms_eac)) 
        top_idx = mean_tfidf.argsort()[-top_k:][::-1]
        keywords = [terms_eac[i] for i in top_idx]

        print(f"\nEAC Cluster {cid}:")
        print("Keywords:", ", ".join(keywords))
else:
    print("無法提取主題：只找到 1 個群集。")"""

#-------------------------------------------------------------------
vec = TfidfVectorizer(max_df=0.4)
X = vec.fit_transform(documents)

from sklearn.cluster import MiniBatchKMeans
mbkm = MiniBatchKMeans(random_state=14, n_clusters=3)

batch_size = 10
for iteration in range(int(X.shape[0] / batch_size)):
 start = batch_size * iteration
 end = batch_size * (iteration + 1)
 mbkm.partial_fit(X[start:end])

labels = mbkm.predict(X)
class PartialFitPipeline(Pipeline):
  def partial_fit(self, X, y=None):
    Xt = X
    for name, transform in self.steps[:-1]:
      Xt = transform.transform(Xt)
    return self.steps[-1][1].partial_fit(Xt, y=y)
  
pipeline = PartialFitPipeline([('feature_extraction',HashingVectorizer()),
 ('clusterer', MiniBatchKMeans(random_state=14, n_clusters=10))])
batch_size = 10
for iteration in range(int(len(documents) / batch_size)):
 start = batch_size * iteration
 end = batch_size * (iteration + 1)
 pipeline.partial_fit(documents[start:end])
labels = pipeline.predict(documents)


labels_online = labels.copy()

sil_online = silhouette_score(X, labels_online)
bc_prec_online = bcubed_precision(labels_online, true_labels)
bc_rec_online = bcubed_recall(labels_online, true_labels)
print("\n=== Number of Clusters (Online Clustering) ===")
print("Online clusters:", len(set(labels_online)))

print("\n=== Online Clustering Evaluation ===")
print("Silhouette Score:", sil_online)
print("BCubed Precision:", bc_prec_online)
print("BCubed Recall:", bc_rec_online)
# ========== b) Online Clustering 找出六大主題 ==========

print("\n===== Online Clustering — Six Main Topics =====")

cluster_counts_online = Counter(labels_online)
top6_online_clusters = [c for c, _ in cluster_counts_online.most_common(6)]
print("Top 6 Online clusters:", top6_online_clusters)

vectorizer_online = vec   
terms_online = vectorizer_online.get_feature_names_out()

for cluster_id in top6_online_clusters:
    mask = (labels_online == cluster_id)
    cluster_matrix = X[mask]  

    mean_tfidf = np.asarray(cluster_matrix.mean(axis=0)).ravel()
    top_indices = mean_tfidf.argsort()[-8:][::-1]
    keywords = [terms_online[i] for i in top_indices]

    print(f"\nOnline Cluster {cluster_id}:")
    print("Keywords:", ", ".join(keywords))
