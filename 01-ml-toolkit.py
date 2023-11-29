from pathlib import Path
import tarfile
import urllib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel


def unzip_and_open_dataset(zip_file_path):
    '''Extract zip file and read into pandas dataframe.'''

    data_directory = Path('datasets') / os.path.splitext(zip_file_path)[0]
    if not data_directory.is_dir():
        data_directory.mkdir(parents=True)
    with zipfile.ZipFile(zip_file_path) as zip_file:
        zip_file.extractall(data_directory)

    csv_file = [file for file in os.listdir(data_directory) if file.endswith('.csv')][0]

    return pd.read_csv(data_directory / csv_file)


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    '''Cluster Similarity custom transformer

    Transformer returns distances of every datapoint to cluster centers'''

    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        X = check_array(X)
        self.n_features_in_ == X.shape[1]
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

