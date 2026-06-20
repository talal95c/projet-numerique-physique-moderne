# Effet tunnel et temps de franchissement d'une barrière de potentiel

Projet numérique en physique moderne — Groupe **MI1_projet_C** (TD groupe 1)
Talal · Thao · Kalim

## Contenu du dépôt

| Fichier | Partie de l'énoncé | Description |
|---|---|---|
| `OndePlane1d1C.py` | Partie 1 — Ondes planes | Génération et superposition d'ondes planes 1D |
| `PaquetOndeGauss1d1C.py` | Partie 2 — Paquets d'ondes | Évolution d'un paquet d'ondes gaussien libre (relation de dispersion de Schrödinger) |
| `DifferencesFinies.py` | Parties 3 et 4 — Résolution numérique + Projet | Solveur de Schrödinger par différences finies : validation sur le cas libre (V₀=0) puis simulation avec barrière de potentiel (effet tunnel) |

## Dépendances

```bash
pip install -r requirements.txt
```

Bibliothèques utilisées : `numpy`, `matplotlib`, `scipy`.

## Ordre d'exécution

Les trois fichiers correspondent à la progression du projet. `DifferencesFinies.py` importe directement `PaquetOndeGauss1d1C.py` (doivent rester dans le même dossier) :

```bash
python3 OndePlane1d1C.py            # Partie 1
python3 PaquetOndeGauss1d1C.py      # Partie 2
python3 DifferencesFinies.py        # Parties 3 et 4 (génère les figures de validation et de simulation)
```

## Résultats de validation

`DifferencesFinies.py` valide le solveur sur le cas de la particule libre (V₀=0) avant de l'utiliser sur la barrière :

- Norme conservée à mieux que **5×10⁻⁸** sur toute la simulation
- Position du maximum conforme à la vitesse de groupe théorique (erreur relative < 1 %)
- Densité de probabilité numérique superposée à la solution analytique

Les figures générées (`validation_V0_0.png`, `simulation_barriere.png`) sont produites automatiquement lors de l'exécution de `DifferencesFinies.py`.

## Auteurs

Code original des parties 1 et 2 : Panayotis Akridas (enseignant). Solveur par différences finies (parties 3-4) : groupe MI1_projet_C.
