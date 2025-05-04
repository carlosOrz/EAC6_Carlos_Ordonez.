import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import plotly
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn import metrics


# create dataset
def create_dataset(number_features):
    """
    Crea un conjunt de dades sintètiques amb 3 clústers mitjançant make_blobs.

    Parameters:
        number_features (int): Nombre d'atributs o característiques que tindran les dades generades.

    Returns:
        tuple: Una tupla (X, y) on:
            - X (ndarray): Matriu de característiques amb les mostres generades.
            - y (ndarray): Etiquetes de clúster per a cada mostra generada.
    """
    
    X, y = make_blobs(
        n_samples=750,  # que es el valor de 250 per 3 blobs
        n_features=number_features,
        centers=3,  # Com hi ha 3 blobs, el valor de n_clusters ha de ser 3
        cluster_std=0.75,  # Desviació estàndard
        shuffle=True,
        random_state=100,
    )
    return X, y

# =========================================================

def plot_data_attributes(X, show=True, nom_alumne="CarlosOrdóñez"):
    """
    Mostra i guarda un gràfic de dispersió dels atributs de les dades en diferents combinacions.

    Parameters:
        X (ndarray): Matriu de característiques amb forma (n_samples, n_features).
        show (bool): Si és True, es mostra el gràfic en pantalla. Per defecte és True.
        nom_alumne (str): Nom de l’alumne que s’utilitzarà per guardar el fitxer PNG.

    Returns:
        None
    """
    
    fig, axs = plt.subplots(2, 2)

    fig.suptitle(f"Data Attributes - {nom_alumne}", fontsize=12, y=1)
    fig.text(0.5, 0.95, "title", horizontalalignment="center")

    fig.set_size_inches(6, 6)

    axs[0, 0].set_title("attr1 vs attr2")
    axs[0, 0].scatter(X[:, 0], X[:, 1], c="white", marker="o", edgecolor="black", s=50)
    axs[0, 1].set_title("attr1 vs attr3")
    axs[0, 1].scatter(X[:, 0], X[:, 2], c="white", marker="o", edgecolor="red", s=50)
    axs[1, 0].set_title("attr1 vs attr4")
    axs[1, 0].scatter(X[:, 0], X[:, 3], c="white", marker="o", edgecolor="blue", s=50)
    axs[1, 1].set_title("attr2 vs attr3")
    axs[1, 1].scatter(X[:, 1], X[:, 2], c="white", marker="o", edgecolor="green", s=50)

    plt.savefig(f"img/scatter_{nom_alumne}.png")
    if show:
        plt.show()

    return None

# =========================================================

def model_kmeans(num_clusters):
    """
    Crea un model de clustering KMeans amb els paràmetres predefinits.

    Parameters:
        num_clusters (int): Nombre de clústers que ha de trobar el model.

    Returns:
        KMeans: Instància del model KMeans configurat però no entrenat.
    """
    
    km = KMeans(
        n_clusters=num_clusters,
        init="random",
        n_init=10,
        max_iter=300,
        tol=1e-04,
        random_state=0,
    )

    return km

# =========================================================

def predict_clusters(model, X):
    """
    Entrena el model de clustering i prediu les etiquetes de clúster per a cada mostra.

    Parameters:
        model (KMeans): Model de clustering (com KMeans).
        X (ndarray): Matriu de dades d'entrada.

    Returns:
        tuple: El model entrenat i les etiquetes de clúster predites per a cada mostra.
    """
    
    y_km = model.fit_predict(X)
    return model, y_km

# =========================================================

def plot_clusters(km, X, y_km, show=True, nom_alumne="CarlosOrdóñez"):
    """
    Dibuixa els clústers generats per KMeans amb tres combinacions d'atributs 
    (attr1 vs attr2, attr1 vs attr3, attr1 vs attr4) i desa la figura com a imatge PNG.

    Parameters:
        km (KMeans): Model KMeans entrenat.
        X (ndarray): Matriu de característiques d'entrada.
        y_km (ndarray): Etiquetes de clúster per a cada mostra.
        show (bool): Si True, mostra el gràfic. Si False, només es desa.
        nom_alumne (str): Nom de l'alumne per personalitzar el nom del fitxer.

    Returns:
        None
    """
    
	# plot the 3 clusters
    fig, axs = plt.subplots(1, 3)

    fig.suptitle(f"Clusters - {nom_alumne}", fontsize=12, y=1)

    fig.set_size_inches(12, 4)

    # attr1 vs attr2
    axs[0].set_title("attr1 vs attr2")
    axs[0].scatter(
        X[y_km == 0, 0],
        X[y_km == 0, 1],
        s=50,
        c="lightgreen",
        marker="s",
        edgecolor="black",
        label="cluster 1",
    )
    axs[0].scatter(
        X[y_km == 1, 0],
        X[y_km == 1, 1],
        s=50,
        c="orange",
        marker="o",
        edgecolor="black",
        label="cluster 2",
    )

    axs[0].scatter(
        X[y_km == 2, 0],
        X[y_km == 2, 1],
        s=50,
        c="lightblue",
        marker="v",
        edgecolor="black",
        label="cluster 3",
    )

    # plot the centroids
    axs[0].scatter(
        km.cluster_centers_[:, 0],
        km.cluster_centers_[:, 1],
        s=250,
        marker="*",
        c="red",
        edgecolor="black",
        label="centroids",
    )

    axs[0].legend(scatterpoints=1)

    # attr1 vs attr3
    axs[1].set_title("attr1 vs attr3")
    axs[1].scatter(
        X[y_km == 0, 0],
        X[y_km == 0, 2],
        s=50,
        c="lightgreen",
        marker="s",
        edgecolor="black",
        label="cluster 1",
    )
    axs[1].scatter(
        X[y_km == 1, 0],
        X[y_km == 1, 2],
        s=50,
        c="orange",
        marker="o",
        edgecolor="black",
        label="cluster 2",
    )

    axs[1].scatter(
        X[y_km == 2, 0],
        X[y_km == 2, 2],
        s=50,
        c="lightblue",
        marker="v",
        edgecolor="black",
        label="cluster 3",
    )

    # plot the centroids
    axs[1].scatter(
        km.cluster_centers_[:, 0],
        km.cluster_centers_[:, 2],
        s=250,
        marker="*",
        c="red",
        edgecolor="black",
        label="centroids",
    )

    axs[1].legend(scatterpoints=1)

    # attr1 vs attr4
    axs[2].set_title("attr1 vs attr4")
    axs[2].scatter(
        X[y_km == 0, 0],
        X[y_km == 0, 3],
        s=50,
        c="lightgreen",
        marker="s",
        edgecolor="black",
        label="cluster 1",
    )
    axs[2].scatter(
        X[y_km == 1, 0],
        X[y_km == 1, 3],
        s=50,
        c="orange",
        marker="o",
        edgecolor="black",
        label="cluster 2",
    )

    axs[2].scatter(
        X[y_km == 2, 0],
        X[y_km == 2, 3],
        s=50,
        c="lightblue",
        marker="v",
        edgecolor="black",
        label="cluster 3",
    )

    # plot the centroids
    axs[2].scatter(
        km.cluster_centers_[:, 0],
        km.cluster_centers_[:, 3],
        s=250,
        marker="*",
        c="red",
        edgecolor="black",
        label="centroids",
    )

    axs[2].legend(scatterpoints=1)

    plt.savefig(f"img/clusters_{nom_alumne}.png")
    if show:
        plt.show()

    return None

# =========================================================

def plot_clusters3D(km, X, y_km, show=True, nom_alumne="CarlosOrdóñez"):
    """
    Dibuixa una representació 3D dels clústers generats pel model KMeans 
    utilitzant els tres primers atributs de les dades. Desa la figura en format PNG.

    Parameters:
        km (KMeans): Model KMeans entrenat.
        X (ndarray): Matriu de característiques d'entrada amb mínim 3 atributs.
        y_km (ndarray): Etiquetes de clúster per a cada mostra.
        show (bool): Si True, mostra el gràfic. Si False, només es desa.
        nom_alumne (str): Nom de l'alumne per incloure al nom del fitxer.

    Returns:
        None
    """

    fig = plt.figure()
    fig.suptitle(f"Clusters 3D - {nom_alumne}", fontsize=12, y=1)
    fig.set_size_inches(6, 6)
    ax = fig.add_subplot(projection="3d")

    # cluster 1
    ax.scatter(
        X[y_km == 0, 0],
        X[y_km == 0, 1],
        X[y_km == 0, 2],
        s=50,
        c="lightgreen",
        marker="s",
        edgecolor="black",
        label="cluster 1",
    )
    # cluster 2
    ax.scatter(
        X[y_km == 1, 0],
        X[y_km == 1, 1],
        X[y_km == 1, 2],
        s=50,
        c="orange",
        marker="o",
        edgecolor="black",
        label="cluster 2",
    )
    # cluster 3
    ax.scatter(
        X[y_km == 2, 0],
        X[y_km == 2, 1],
        X[y_km == 2, 2],
        s=50,
        c="lightblue",
        marker="v",
        edgecolor="black",
        label="cluster 3",
    )
    # plot the centroids
    ax.scatter(
        km.cluster_centers_[:, 0],
        km.cluster_centers_[:, 1],
        km.cluster_centers_[:, 2],
        s=250,
        marker="*",
        c="red",
        edgecolor="black",
        label="centroids",
    )

    ax.set_xlabel("attr1")
    ax.set_ylabel("attr2")
    ax.set_zlabel("attr3")

    ax.legend(scatterpoints=1)
    ax.grid()
    plt.savefig(f"img/clusters3D_{nom_alumne}.png")
    if show:
        plt.show()

    return None

# =========================================================

def plot_clusters3D_HTML(X, y_km, show=True, nom_alumne="CarlosOrdóñez"):
    """
    Genera una representació interactiva en 3D dels clústers mitjançant Plotly 
    i desa el resultat com una pàgina HTML navegable.

    Parameters:
        X (ndarray): Matriu de característiques d'entrada amb mínim 3 atributs.
        y_km (ndarray): Etiquetes de clúster per a cada mostra.
        show (bool): Si True, obre automàticament la pàgina HTML generada.
        nom_alumne (str): Nom de l'alumne per a personalitzar el nom del fitxer HTML.

    Returns:
        None
    """

    # Configure the trace.
    # cluster 1
    cluster1 = go.Scatter3d(
        x=X[y_km == 0, 0],
        y=X[y_km == 0, 1],
        z=X[y_km == 0, 2],
        mode="markers",
        marker={"size": 3, "opacity": 0.8, "color": "red"},
        name="cluster 1",
    )
    # cluster 2
    cluster2 = go.Scatter3d(
        x=X[y_km == 1, 0],
        y=X[y_km == 1, 1],
        z=X[y_km == 1, 2],
        mode="markers",
        marker={"size": 3, "opacity": 0.8, "color": "blue"},
        name="cluster 2",
    )
    # cluster 3
    cluster3 = go.Scatter3d(
        x=X[y_km == 2, 0],
        y=X[y_km == 2, 1],
        z=X[y_km == 2, 2],
        mode="markers",
        marker={"size": 3, "opacity": 0.8, "color": "green"},
        name="cluster 3",
    )

    # Configure the layout.
    layout = go.Layout(
        title=f"Clusters 3D - {nom_alumne}", margin={"l": 0, "r": 0, "b": 0, "t": 50}
    )

    data = [cluster1, cluster2, cluster3]
    plot_figure = go.Figure(data=data, layout=layout)

    plotly.offline.plot(
        plot_figure, filename=f"img/clusters3D_HTML_{nom_alumne}.html", auto_open=show
    )

    return None

# =========================================================

def transform_PCA(X, num_components):
    """
    Aplica una reducció de dimensionalitat a les dades mitjançant PCA (Anàlisi de components principals).

    Parameters:
        X (ndarray): Matriu de característiques d'entrada.
        num_components (int): Nombre de components principals a conservar.

    Returns:
        ndarray: Matriu transformada amb el nombre especificat de components principals.
    """
    
    X_PCA = PCA(n_components=num_components).fit_transform(X)
    return X_PCA

# =========================================================

def plot_elbow(X_PCA, show=True, nom_alumne="CarlosOrdóñez"):
    """
    Dibuixa la gràfica del mètode del colze per determinar el nombre òptim de clústers.

    Parameters:
        X_PCA (ndarray): Matriu de dades reduïdes amb PCA.
        show (bool): Si True, mostra la gràfica en pantalla. Si False, només la desa.
        nom_alumne (str): Nom de l'alumne, utilitzat per personalitzar el títol i el nom del fitxer.

    Returns:
        None
    """

    distortions = []
    K = range(1, 10)

    for k in K:
        km = KMeans(
            n_clusters=k,
            init="random",
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0,
        )
        km.fit(X_PCA)
        distortions.append(km.inertia_)

    plt.figure()
    plt.plot(K, distortions, marker="o")
    plt.xlabel("Número de clústers")
    plt.ylabel("Distorsió")
    plt.title(f"Elbow Method - {nom_alumne}")
    plt.savefig(f"img/elbow_{nom_alumne}.png")
    if show:
        plt.show()

    return None

# =========================================================

def plot_clusters_PCA(km, X, y_km, show=True, nom_alumne="CarlosOrdóñez", pca=False):
    """
    Representa gràficament els clústers resultants després d'aplicar PCA (2D) juntament amb els centroides.

    Parameters:
        km (KMeans): Model KMeans entrenat.
        X (ndarray): Dades reduïdes a dues dimensions amb PCA.
        y_km (ndarray): Assignació de clústers per a cada mostra.
        show (bool): Si True, mostra la gràfica. Si False, només la desa.
        nom_alumne (str): Nom de l'alumne, utilitzat per personalitzar el nom del fitxer i el títol.
        pca (bool): Paràmetre sense ús funcional, només per compatibilitat amb altres funcions.

    Returns:
        None
    """

    # Definim titol segons si s'ha aplicat PCA o no:
    titol = f"Clusters PCA - {nom_alumne}"

    # Dibuixem només 1 gràfica: attr1 vs attr2 (dues dimensions després de PCA)
    plt.figure()
    plt.title(titol)
    for i, color, marker in zip(
        range(3), ["lightgreen", "orange", "lightblue"], ["s", "o", "v"]
    ):
        plt.scatter(
            X[y_km == i, 0],
            X[y_km == i, 1],
            s=50,
            c=color,
            marker=marker,
            edgecolor="black",
            label=f"cluster {i+1}",
        )
    plt.scatter(
        km.cluster_centers_[:, 0],
        km.cluster_centers_[:, 1],
        s=250,
        marker="*",
        c="red",
        edgecolor="black",
        label="centroids",
    )
    plt.legend()

    plt.savefig(f"img/clusters_PCA_{nom_alumne}.png")
    if show:
        plt.show()

    return None

# =========================================================

def calcular_scores(y_true, y_pred):
    """
    Calcula l'homogeneïtat i la completesa entre etiquetes reals i predites.

    Parameters:
        y_true (array-like): Etiquetes reals.
        y_pred (array-like): Etiquetes predites pel model.

    Returns:
        tuple: (homogeneïtat, completesa)
    """
    
    from sklearn.metrics import homogeneity_score, completeness_score

    h = homogeneity_score(y_true, y_pred)
    c = completeness_score(y_true, y_pred)
    return h, c

# =========================================================