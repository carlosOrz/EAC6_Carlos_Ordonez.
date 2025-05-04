import unittest
from functions import create_dataset, transform_PCA, model_kmeans, predict_clusters


class TestClustering(unittest.TestCase):

    def setUp(self):
        """Prepare data and models before each test."""
        self.X, self.y = create_dataset(4)
        self.X_pca = transform_PCA(self.X, 2)

        self.km = model_kmeans(3)
        self.km_pca = model_kmeans(3)

        _, self.y_km = predict_clusters(self.km, self.X)
        _, self.y_km_pca = predict_clusters(self.km_pca, self.X_pca)

    def test_num_atributs_originals(self):
        """Comprova que les dades originals tenen 4 atributs"""
        self.assertEqual(self.X.shape[1], 4)

    def test_num_atributs_pca(self):
        """Comprova que les dades transformades amb PCA tenen 2 atributs"""
        self.assertEqual(self.X_pca.shape[1], 2)

    def test_clusters_assignacio_igual(self):
        """Comprova que les assignacions dels clústers són iguals"""
        self.assertListEqual(self.y_km.tolist(), self.y_km_pca.tolist())


if __name__ == '__main__':
    unittest.main()
