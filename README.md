# Projecte EAC6 - Clustering amb PCA i Visualització

## Estructura del projecte

```
El projecte es compon dels següents elements principals:

    main.py: el fitxer principal que executa tot el flux del projecte (generació de dades, clustering, visualitzacions, PCA, etc.).

    functions.py: conté totes les funcions auxiliars necessàries per al tractament, clustering i visualització de dades.

    img/: carpeta on es guarden automàticament totes les gràfiques generades (imatges PNG i fitxers HTML).

    tests/: conté la suite de tests (testsEAC6.py) per verificar el correcte funcionament de parts clau del projecte.

    doc/: carpeta amb la documentació generada automàticament amb pydoc.

    requirements.txt: llista de totes les dependències necessàries per executar el projecte.

    LICENCE: fitxer amb la llicència del projecte.

    README.md: aquest mateix fitxer amb informació sobre estructura, ús i execució.
```

## Instal·lació del projecte

1. Clona el repositori o descarrega el codi
2. Instal·la les dependències
   ```bash
   pip install -r requirements.txt
   ```

## Execució del projecte

Per executar el projecte i generar totes les visualitzacions:

```bash
python main.py
```

Les imatges es guardaran automàticament dins de la carpeta `img/`.

## Comprovació de l’anàlisi estàtic (Pylint)

Per comprovar la qualitat del codi amb `pylint`, executa:

```bash
pylint main.py functions.py
```

Assegura't de tenir `pylint` instal·lat:

```bash
pip install pylint
```

## Generació de la documentació

Per generar la documentació amb `pydoc`:

```bash
pydoc -w functions
```

Això generarà `functions.html`, que pots moure a la carpeta `doc/`.

## Comprovació dels tests

Per executar els tests automàtics amb `unittest`:

```bash
python -m unittest tests/testsEAC6.py
```

Tots els tests haurien de passar correctament (`OK`).
