"""
main script
"""

from functions import (
    create_dataset,
    plot_data_attributes,
    model_kmeans,
    predict_clusters,
    plot_clusters,
    plot_clusters3D,
    plot_clusters3D_HTML,
    transform_PCA,
    plot_elbow,
    plot_clusters_PCA,
    calcular_scores,
)

# creem el dataset
X, y = create_dataset(4)

# tenim 4 atributs
# Número d'atributs de les dades
print(f"Número d'atributs: {X.shape[1]}")

# Els 5 primers elements de l'atribut 1
print("5 primers elements de l'atribut 1:", X[:5, 0])

# printem les dades, amb els atributs en els diferents eixos
plot_data_attributes(X)

# model de clúster amb KMeans
km = model_kmeans(3)
# entrenament del model
km, y_km = predict_clusters(km, X)

# grafiquem els clústers amb diferents formats:
# a) format 2D
print("Format 2D")
plot_clusters(km, X, y_km)

# b) format 3D
print("Format 3D")
plot_clusters3D(km, X, y_km)

# c) format 3D amb HTML
print("Format 3D HTML")
plot_clusters3D_HTML(X, y_km)

# cada clúster té els 4 atributs
# Printem les 5 primeres dades de cada clúster
for i in range(3):
    print(f"\n5 primers elements del clúster {i + 1}:")
    print(X[y_km == i][:5])

# PCA ============================
# transformem les dades a dos atributs
X_PCA = transform_PCA(X, 2)

# Comprovar que després de la transformació PCA el número d'atributs és 2
print(f"\nNúmero d'atributs després de PCA: {X_PCA.shape[1]}")

# Mètode del colze per comprovar que n=3 és una bona opció
print("Gràfica Elbow")
plot_elbow(X_PCA)

# Model KMeans amb PCA
km_PCA = model_kmeans(3)
km_PCA, y_km_PCA = predict_clusters(km_PCA, X_PCA)
print(y_km_PCA[:10])

# Grafiquem els clústers
print("Gràfica Clusters PCA")
plot_clusters_PCA(
    km_PCA, X_PCA, y_km_PCA, show=True, nom_alumne="CarlosOrdóñez", pca=True
)

# Comparació dels resultats (assignació dels clústers) sense i amb PCA
print("\nComparació d’assignacions:")
print("Sense PCA:", y_km)
print("Amb PCA:  ", y_km_PCA)

# Comprovar que aquestes dues llistes de valors són iguals
print("\nLes assignacions de clústers són iguals?", y_km.tolist() == y_km_PCA.tolist())

h1, c1 = calcular_scores(y, y_km)
h2, c2 = calcular_scores(y, y_km_PCA)

print(f"\n[Original] Homogeneïtat: {h1:.3f} | Completesa: {c1:.3f}")
print(f"[PCA]      Homogeneïtat: {h2:.3f} | Completesa: {c2:.3f}")
