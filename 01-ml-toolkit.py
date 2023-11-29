from pathlib import Path
import tarfile
import urllib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

def load_housing_data(file_name, url):
    '''Function to automatically download decompress tarball file and read in data frame'''

    dataset_directory = os.path.splitext(file_name)[0]
    tarball_path = Path('datasets') / dataset_directory / file_name
    if not tarball_path.is_file():
        Path('datasets').mkdir(parents=True, exist_ok=True)
        url = url
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as file_tarball:
            file_tarball.extractall(path = Path('datasets') / dataset_directory)

    dataset_csv = [file for file in os.listdir(Path('datasets') / dataset_directory) if file.endswith('.csv')][0]

    return pd.read_csv(Path('datasets') / dataset_directory / dataset_csv)


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

