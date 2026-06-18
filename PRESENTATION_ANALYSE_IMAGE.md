# 🔬 PRÉSENTATION ANALYSE D'IMAGE — PARCOURS UTILISATEUR COMPLET

**Application :** Détection de Contours, Lissage & Suppression d'Arrière-plan  
**Date :** 12 Juin 2026  
**Technologie :** Python + Streamlit + OpenCV + NumPy + SciPy  
**Type de présentation :** Guide d'utilisation étape par étape avec captures d'écran


---

# APERÇU GÉNÉRAL DE L'APPLICATION

## Qu'est-ce que cette application ?

Cette application est un **outil complet d'analyse numérique d'images** qui implémente, depuis zéro ou via des bibliothèques spécialisées, les grandes familles d'algorithmes du traitement d'image :

- **Lissage et amélioration** — Préparation du signal image (Gaussien, Bilatéral, Médian, CLAHE, Splines)
- **Détection de contours** — Dérivées discrètes et transformée de Fourier (Sobel+NMS, Canny, LoG, FFT Butterworth)
- **Segmentation et suppression de fond** — Algorithmes classiques et Intelligence Artificielle (GrabCut+Saillance, Flood Fill, K-means LAB, **rembg U²-Net**)
- **Composition avancée** — Résolution d'équations de Poisson pour du photomontage seamless
- **Post-traitement visuel** — Watermarking, retouche, effets artistiques, ombres et reflets

L'application fonctionne entièrement dans le navigateur via **Streamlit** et peut être utilisée par des photographes, graphistes, développeurs, ou toute personne souhaitant traiter des images sans logiciel complexe.

## Architecture Technique

```
┌─────────────────────────────────────────────────────────────────────┐
│                   ANALYSE D'IMAGE — ARCHITECTURE                    │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    INTERFACE STREAMLIT                        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │   │
│  │  │ Vue      │  │ Vue      │  │ Compar.  │  │ Suppression  │  │   │
│  │  │ Unique   │  │ Complète │  │ A vs B   │  │ Fond         │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │   │
│  │  ┌──────────────┐  ┌──────────────────────────────────────┐  │   │
│  │  │ Composition  │  │ Avant/Après (curseur glissant)       │  │   │
│  │  │ & Poisson    │  │                                      │  │   │
│  │  └──────────────┘  └──────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│           │                                                         │
│           ▼                                                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │               MOTEUR DE TRAITEMENT (Python)                   │   │
│  │                                                               │   │
│  │  PRÉ-TRAITEMENT      DÉTECTION CONTOURS    SUPPRESSION FOND   │   │
│  │  ┌──────────────┐   ┌──────────────┐     ┌──────────────┐    │   │
│  │  │ CLAHE        │   │ Sobel + NMS  │     │ GrabCut      │    │   │
│  │  │ Gaussien     │   │ Canny        │     │ Flood Fill   │    │   │
│  │  │ Bilatéral    │   │ LoG (Zéro.)  │     │ K-means LAB  │    │   │
│  │  │ Médian       │   │ Spline Edges │     │ rembg (IA)   │    │   │
│  │  │ Spline Lissée│   │ FFT HP/BP    │     │              │    │   │
│  │  └──────────────┘   └──────────────┘     └──────────────┘    │   │
│  │                                                               │   │
│  │  RAFFINEMENT MASQUE    POISSON BLENDING    POST-TRAITEMENT    │   │
│  │  ┌──────────────┐   ┌──────────────┐     ┌──────────────┐    │   │
│  │  │ Fill Holes   │   │ Normal       │     │ Watermark    │    │   │
│  │  │ Remove Isl.  │   │ Gradient Max │     │ Reflet       │    │   │
│  │  │ Edge Snap    │   │ Gradient Min │     │ Ombre        │    │   │
│  │  │ Alpha Matting│   │              │     │ Sketch/B.P.  │    │   │
│  │  └──────────────┘   └──────────────┘     └──────────────┘    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│           │                                                         │
│           ▼                                                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              BIBLIOTHÈQUES SCIENTIFIQUES                      │   │
│  │  OpenCV │ NumPy │ SciPy │ Matplotlib │ Pillow │ rembg+ONNX   │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Les 3 Grands Cas d'Usage

L'application couvre trois besoins distincts que l'utilisateur peut combiner librement :

### 🔍 1. ANALYSE DE CONTOURS
Visualiser les **bords** et les **structures** d'une image via des algorithmes mathématiques variés. Idéal pour l'analyse scientifique, la comparaison de méthodes, ou l'enseignement du traitement d'image.
- **13 méthodes** de lissage et détection de contours
- Comparaison côte-à-côte avec carte de différence
- Visualisation spectrale (FFT)

### ✂️ 2. SUPPRESSION D'ARRIÈRE-PLAN (DÉTOURAGE)
Isoler le sujet principal d'une image et le placer sur un fond transparent, blanc, ou de couleur. C'est l'équivalent d'un **détourage automatique intelligent**.
- **4 algorithmes** : GrabCut+Saillance, Flood Fill, K-means LAB, rembg IA
- **16 modèles IA** disponibles avec rembg
- Raffinement automatique du masque (trous, îlots, bords)
- Alpha matting pour des bords doux et naturels

### 🎨 3. COMPOSITION & PHOTOMONTAGE
Incruster un sujet détouré dans un nouveau décor avec un rendu **ultra-réaliste** grâce aux équations de Poisson. C'est la technologie derrière les outils comme Photoshop "Collage seamless".
- **3 modes de fusion** : Normal, Gradient Max, Gradient Min
- Transformation interactive (échelle, rotation, position)
- Harmonisation automatique des couleurs
- Ombres portées réalistes

## Algorithmes disponibles

### Détection de Contours et Lissage (13 méthodes)

| # | Méthode | Type | Description |
|---|---------|------|-------------|
| 1 | 🖼 Original | — | Image d'origine sans traitement |
| 2 | ✨ CLAHE | Lissage | Égalisation adaptative d'histogramme — améliore le contraste local |
| 3 | 🔵 Gaussien | Lissage | Filtre gaussien — atténue le bruit, floute l'image |
| 4 | 🟠 Bilatéral | Lissage | Préserve les bords tout en lissant les surfaces |
| 5 | 🟡 Médian | Lissage | Supprime le bruit impulsionnel (poivre et sel) |
| 6 | 🟢 Spline Lissée | Lissage | Reconstruction par splines bicubiques après sous-échantillonnage |
| 7 | 🔴 Sobel + NMS | Contour | Gradient + Suppression Non-Maximale — contours fins 1 pixel |
| 8 | ⚡ Canny | Contour | Algorithme de Canny complet (double seuillage + hystérèse) |
| 9 | 🔮 LoG Zéro-croisements | Contour | Laplacien du Gaussien — localisation précise par passages à zéro |
| 10 | 🌿 Contours Spline | Contour | Gradient sur image reconstruite par splines |
| 11 | 🟣 FFT Spectre | Fréquentiel | Spectre d'amplitude (log-scale) — visualisation des fréquences |
| 12 | 📈 FFT Passe-haut | Fréquentiel | Filtre Butterworth passe-haut — isole les hautes fréquences (bords) |
| 13 | 📉 FFT Passe-bas | Fréquentiel | Filtre Butterworth passe-bas — isole les basses fréquences (fond) |

### Suppression d'Arrière-plan (4 algorithmes)

| # | Méthode | Principe | Qualité | Vitesse |
|---|---------|----------|---------|---------|
| 1 | **rembg (IA U²-Net)** | Réseau de neurones profond (Deep Learning) | ⭐⭐⭐⭐ | Lente |
| 2 | **GrabCut + Saillance** | Modèle probabiliste (GMM + MRF) + saillance spectrale | ⭐⭐⭐ | Moyenne |
| 3 | **Flood Fill Bordures** | Remplissage par diffusion depuis les bords | ⭐⭐ | Rapide |
| 4 | **K-means LAB** | Clustering dans l'espace CIELAB + coordonnées spatiales | ⭐⭐ | Rapide |

## Modèles IA disponibles (rembg)

L'application supporte **16 modèles de deep learning** pour la suppression de fond :

| Modèle | Usage | Taille |
|--------|-------|--------|
| `u2net` | Usage général | Standard |
| `u2netp` | Usage général (léger) | Petit |
| `u2net_human_seg` | Segmentation de personnes | Standard |
| `u2net_cloth_seg` | Segmentation de vêtements | Standard |
| `silueta` | Usage général | Standard |
| `isnet-general-use` | Usage général haute qualité | Grand |
| `isnet-anime` | Images animées/manga | Grand |
| `birefnet-general` | Usage général (BiRefNet) | Grand |
| `birefnet-general-lite` | Usage général léger (BiRefNet) | Moyen |
| `birefnet-portrait` | Portraits (BiRefNet) | Grand |
| `birefnet-dis` | Segmentation dichotomique | Grand |
| `birefnet-hrsod` | Détection objets saillants HR | Grand |
| `birefnet-cod` | Détection objets camouflés | Grand |
| `birefnet-massive` | Qualité maximale | Très grand |
| `sam` | Segment Anything Model (Meta) | Grand |
| `ben2-base` | Background Erasing Network v2 | Standard |

## Pipeline de Traitement

```
IMAGE CHARGÉE
      │
      ▼
┌─────────────────────────────────────────────────┐
│ 1. DÉCODAGE                                     │
│    Format → Niveaux de gris (float64) + RGB    │
└─────────────────────────────────────────────────┘
      │
      ├────────────────────┬──────────────────────────┐
      ▼                    ▼                          ▼
┌──────────────┐  ┌────────────────┐  ┌──────────────────────┐
│ 2a. LISSAGE  │  │ 2b. DÉTECTION  │  │ 2c. SUPPRESSION FOND │
│ CLAHE        │  │    CONTOURS    │  │                      │
│ Gaussien     │  │ Sobel+NMS      │  │ Choix algorithme :   │
│ Bilatéral    │  │ Canny          │  │ • GrabCut+Saillance  │
│ Médian       │  │ LoG            │  │ • Flood Fill         │
│ Spline       │  │ Spline Edges   │  │ • K-means LAB        │
│              │  │ FFT HP/BP      │  │ • rembg IA (U²-Net)  │
└──────────────┘  └────────────────┘  └──────────┬───────────┘
      │                    │                      │
      ▼                    ▼                      ▼
┌──────────────┐  ┌────────────────┐  ┌──────────────────────────┐
│ Visualisation│  │ Visualisation  │  │ 3. RAFFINEMENT DU MASQUE │
│ (Onglet Vue  │  │ (Onglets Vue   │  │ • Fermeture morphologique│
│  Unique,     │  │  Unique, Vue   │  │ • Fill Holes (trous)     │
│  Complète,   │  │  Complète,     │  │ • Remove Islands (bruit) │
│  Comparaison)│  │  Comparaison)  │  │ • Edge Snap (Canny)      │
└──────────────┘  └────────────────┘  │ • Alpha Matting (bords   │
                                       │   doux)                  │
                                       └──────────┬───────────────┘
                                                  │
                                                  ▼
                                       ┌──────────────────────────┐
                                       │ 4. APPLICATION DU MASQUE │
                                       │ • Fond transparent (PNG)  │
                                       │ • Fond blanc              │
                                       │ • Fond couleur au choix   │
                                       └──────────┬───────────────┘
                                                  │
                                                  ▼
                                       ┌──────────────────────────┐
                                       │ 5. POST-TRAITEMENT       │
                                       │ • Filtres artistiques     │
                                       │   (Sketch, Blueprint)     │
                                       │ • Reflet miroir           │
                                       │ • Ajustements L/C/S/N     │
                                       │ • Recadrage auto          │
                                       │ • Watermark (filigrane)   │
                                       └──────────┬───────────────┘
                                                  │
                                                  ▼
                                       ┌──────────────────────────┐
                                       │ 6. COMPOSITION POISSON   │
                                       │ (optionnel)               │
                                       │ • Transformation          │
                                       │ • Harmonisation couleur   │
                                       │ • Ombre portée            │
                                       │ • Fusion seamless         │
                                       └──────────┬───────────────┘
                                                  │
                                                  ▼
                                       ┌──────────────────────────┐
                                       │ 7. EXPORT                │
                                       │ • PNG transparent         │
                                       │ • PNG fond coloré/blanc  │
                                       │ • ZIP (traitement lot)   │
                                       └──────────────────────────┘
```

## Chiffres Clés du Projet

| Métrique | Valeur |
|----------|--------|
| **Fichier principal** | 1 (`code.py` — 1 300 lignes) |
| **Algorithmes implémentés** | 25+ |
| **Méthodes de détection/lissage** | 13 |
| **Algorithmes de suppression de fond** | 4 |
| **Modèles IA disponibles** | 16 |
| **Modes de fusion Poisson** | 3 |
| **Onglets Streamlit** | 6 |
| **Paramètres utilisateur ajustables** | 30+ |
| **Formats d'image supportés** | 6 (PNG, JPG, JPEG, BMP, TIFF, WEBP) |
| **Bibliothèques Python** | 7+ (Streamlit, OpenCV, NumPy, SciPy, Matplotlib, Pillow, rembg) |
| **Fondements mathématiques** | Équation de Poisson, FFT, Butterworth, Splines Bicubiques, GMM, MRF, K-means, Distance Transform |

---

> **📌 Note importante :** Cette présentation suit le parcours complet d'un utilisateur à travers toutes les fonctionnalités de l'application. Chaque section contient des emplacements `[📷 CAPTURE: ...]` où vous devez insérer vos captures d'écran.

---

# TABLE DES MATIÈRES

1. [ÉTAPE 1 — Installation et Lancement](#étape-1--installation-et-lancement)
2. [ÉTAPE 2 — Interface Générale et Barre Latérale](#étape-2--interface-générale-et-barre-latérale)
3. [ÉTAPE 3 — Charger une ou plusieurs Images](#étape-3--charger-une-ou-plusieurs-images)
4. [ÉTAPE 4 — Paramètres de Lissage et Détection](#étape-4--paramètres-de-lissage-et-détection)
5. [ÉTAPE 5 — Onglet Vue Unique (Analyse individuelle)](#étape-5--onglet-vue-unique-analyse-individuelle)
6. [ÉTAPE 6 — Onglet Vue Complète (Grille de toutes les méthodes)](#étape-6--onglet-vue-complète-grille-de-toutes-les-méthodes)
7. [ÉTAPE 7 — Onglet Comparaison (Côte à côte)](#étape-7--onglet-comparaison-côte-à-côte)
8. [ÉTAPE 8 — Onglet Suppression Fond (Détourage)](#étape-8--onglet-suppression-fond-détourage)
9. [ÉTAPE 9 — Paramètres Avancés de Suppression de Fond](#étape-9--paramètres-avancés-de-suppression-de-fond)
10. [ÉTAPE 10 — Raffinement du Masque](#étape-10--raffinement-du-masque)
11. [ÉTAPE 11 — Fonctions de Post-Traitement](#étape-11--fonctions-de-post-traitement)
12. [ÉTAPE 12 — Incrustation Poisson (Photomontage)](#étape-12--incrustation-poisson-photomontage)
13. [ÉTAPE 13 — Traitement par Lot (Export ZIP)](#étape-13--traitement-par-lot-export-zip)
14. [ÉTAPE 14 — Onglet Avant/Après](#étape-14--onglet-avantapres)
15. [ÉTAPE 15 — Filtres Artistiques et Effets Visuels](#étape-15--filtres-artistiques-et-effets-visuels)
16. [ÉTAPE 16 — Filigrane (Watermark)](#étape-16--filigrane-watermark)

---

# ÉTAPE 1 — Installation et Lancement

➤ **Prérequis :** Python 3.10+ installé  
➤ **Commande de lancement :** `streamlit run code.py`

## 1.1 Installation des dépendances

Avant la première utilisation, installez les bibliothèques requises :

```bash
# Dépendances obligatoires
pip install streamlit numpy opencv-python scipy matplotlib pillow

# Dépendance optionnelle (IA — meilleure qualité de détourage)
pip install rembg onnxruntime
```

> 💡 **Sans rembg**, toutes les fonctionnalités restent accessibles (GrabCut, Flood Fill, K-means). rembg ajoute simplement la détection par **Intelligence Artificielle (U²-Net)**.

## 1.2 Lancement de l'application

Ouvrez un terminal dans le dossier du projet et exécutez :

```bash
streamlit run code.py
```

Streamlit démarre un serveur local et ouvre automatiquement votre navigateur par défaut.

```
┌──────────────────────────────────────────────────────────────┐
│                    TERMINAL                                  │
│                                                              │
│  $ streamlit run code.py                                    │
│                                                              │
│  You can now view your Streamlit app in your browser.       │
│                                                              │
│  Local URL:            http://localhost:8501                 │
│  Network URL:          http://192.168.1.10:8501              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

> [📷 CAPTURE 1A : **Terminal après lancement** — Afficher la commande et l'URL locale]

> [📷 CAPTURE 1B : **Navigateur à l'ouverture** — Page d'accueil avant chargement d'image, avec le message "Chargez une image dans le panneau gauche"]

---

# ÉTAPE 2 — Interface Générale et Barre Latérale

➤ **Interface :** Layout Streamlit avec barre latérale + zone principale à 6 onglets

## 2.1 Structure de l'Interface

```
┌──────────────────────────┬──────────────────────────────────────────┐
│     BARRE LATÉRALE       │        ZONE PRINCIPALE                    │
│     (Panneau gauche)     │        (Centre de l'écran)                 │
│                          │                                           │
│  ⚙️ PARAMÈTRES           │  🔬 Détection, Lissage & Suppression      │
│                          │     d'Arrière-plan                        │
│  📁 Image(s)             │                                           │
│  [Browse files]          │  ┌─────────────────────────────────────┐  │
│                          │  │ 🔍 Vue    │ 🖼 Vue   │ ⚖️ Compar.  │  │
│  🖼 Image à traiter ▼    │  │  unique   │ complète │             │  │
│                          │  ├──────────┴──────────┴─────────────┤  │
│  🔵 Gaussien / LoG       │  │ ✂️ Suppr. │ 🎨 Comp.  │ 🌓 Avant/ │  │
│  σ sigma  [───●──] 1.5  │  │   Fond    │ & Poisson │   Après   │  │
│                          │  └─────────────────────────────────────┘  │
│  ⚡ Canny                │                                           │
│  Seuil bas   [─●───] 0.05│  ┌─────────────────────────────────────┐  │
│  Seuil haut  [──●──] 0.18│  │                                     │  │
│                          │  │    CONTENU DE L'ONGLET ACTIF        │  │
│  🟠 Bilatéral            │  │                                     │  │
│  σ couleur  [─●───] 0.10│  │    (Images, graphiques, contrôles)   │  │
│  σ spatial  [──●──] 10.0│  │                                     │  │
│                          │  └─────────────────────────────────────┘  │
│  🟡 Médian               │                                           │
│  Taille noyau [●───] 5  │                                           │
│                          │                                           │
│  🟣 FFT Butterworth      │                                           │
│  Cutoff   [─●───] 0.08   │                                           │
│  Ordre    [──●──] 2      │                                           │
│                          │                                           │
│  🟢 Spline               │                                           │
│  Échelle  [──●──] 0.5    │                                           │
│                          │                                           │
│  ✂️ Suppression Arrière-plan                                        │
│  Méthode  [rembg IA  ▼]  │                                           │
│                          │                                           │
│  [▶ Appliquer] (bouton)  │                                           │
│                          │                                           │
│  🐟 Incrustation Poisson │                                           │
│  🖼 Image Cible (Fond)   │                                           │
│  [Browse files]          │                                           │
│                          │                                           │
│  🖋️ Filigrane (Watermark)│                                           │
│  Texte [___________]     │                                           │
│                          │                                           │
│  🌟 Fonctions Avancées   │                                           │
│  [ ] Recadrage auto      │                                           │
│  [ ] Reflet miroir       │                                           │
│  Filtre artistique  ▼    │                                           │
│  Luminosité [──●──] 0    │                                           │
│  Contraste  [──●──] 0    │                                           │
│  Saturation [──●──] 1.0  │                                           │
│  Netteté    [──●──] 0    │                                           │
│                          │                                           │
│  🎨 Colormap             │                                           │
│  [Auto ▼]                │                                           │
└──────────────────────────┴──────────────────────────────────────────┘
```

> [📷 CAPTURE 2A : **Interface complète au lancement** — Vue d'ensemble avec barre latérale et zone principale]

> [📷 CAPTURE 2B : **Barre latérale complète** — Tous les paramètres visibles en scrollant]

---

# ÉTAPE 3 — Charger une ou plusieurs Images

➤ **Emplacement :** Barre latérale → Section "⚙️ Paramètres" → "📁 Image(s)"  
➤ **Formats :** PNG, JPG, JPEG, BMP, TIFF, WEBP

## 3.1 Chargement d'une image unique

1. Cliquez sur le bouton **"Browse files"** dans la zone "📁 Image(s)"
2. Sélectionnez un fichier image sur votre ordinateur
3. L'image est immédiatement décodée et affichée

Dès le chargement, l'application affiche :
- **Largeur** en pixels
- **Hauteur** en pixels
- **Taille** en kilopixels (Kpx)

## 3.2 Chargement de plusieurs images

Si vous sélectionnez **plusieurs fichiers** :
- Un menu déroulant **"🖼 Image à traiter"** apparaît
- Vous pouvez basculer entre les images sans les recharger
- L'option **"Traitement par lot"** devient disponible (génération d'un ZIP)

> [📷 CAPTURE 3A : **Interface après chargement d'une image** — Métriques largeur/hauteur/taille affichées]

> [📷 CAPTURE 3B : **Chargement multiple** — Menu déroulant "Image à traiter" visible avec plusieurs fichiers]

---

# ÉTAPE 4 — Paramètres de Lissage et Détection

➤ **Emplacement :** Barre latérale — Sections dédiées à chaque algorithme

## 4.1 Paramètres disponibles

### 🔵 Gaussien / LoG
| Paramètre | Plage | Défaut | Effet |
|-----------|-------|--------|-------|
| **σ sigma** | 0.3 – 6.0 | 1.5 | Intensité du flou gaussien — ↑ = plus flou, moins de bruit |

### ⚡ Canny
| Paramètre | Plage | Défaut | Effet |
|-----------|-------|--------|-------|
| **Seuil bas** | 0.01 – 0.30 | 0.05 | Sensibilité aux contours faibles |
| **Seuil haut** | 0.05 – 0.60 | 0.18 | Seuil pour les contours forts |

> 💡 **Astuce Canny :** Augmentez le seuil bas pour moins de contours parasites. Baissez-le pour capturer plus de détails.

### 🟠 Bilatéral
| Paramètre | Plage | Défaut | Effet |
|-----------|-------|--------|-------|
| **σ couleur** | 0.01 – 0.50 | 0.10 | Similarité de couleur pour le lissage |
| **σ spatial** | 2.0 – 30.0 | 10.0 | Portée spatiale du filtre |

### 🟡 Médian
| Paramètre | Plage | Défaut | Effet |
|-----------|-------|--------|-------|
| **Taille noyau** | 3 – 15 | 5 | Taille de la fenêtre (impaire forcée) |

### 🟣 FFT Butterworth
| Paramètre | Plage | Défaut | Effet |
|-----------|-------|--------|-------|
| **Cutoff** | 0.02 – 0.40 | 0.08 | Fréquence de coupure du filtre |
| **Ordre** | 1 – 6 | 2 | Pente du filtre (ordre du polynôme) |

### 🟢 Spline
| Paramètre | Plage | Défaut | Effet |
|-----------|-------|--------|-------|
| **Échelle** | 0.1 – 1.0 | 0.5 | Taux de sous-échantillonnage avant reconstruction |

> [📷 CAPTURE 4A : **Section lissage dans la barre latérale** — Paramètres Gaussien, Bilatéral, Médian, Spline visibles]

> [📷 CAPTURE 4B : **Section Canny** — Curseurs seuil bas et seuil haut]

> [📷 CAPTURE 4C : **Section FFT Butterworth** — Curseurs Cutoff et Ordre]

---

# ÉTAPE 5 — Onglet Vue Unique (Analyse Individuelle)

➤ **Onglet :** 🔍 Vue unique  
➤ **Usage :** Analyser une méthode à la fois, en grand format

## 5.1 Interface

```
┌──────────────────────────────────────────────────────────────┐
│  [🔍 Vue unique] [🖼 Vue complète] [⚖️ Comparaison] …       │
│                                                              │
│  Méthode : [🔴 Sobel + NMS  ▼]  ← 13 méthodes disponibles   │
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │                                                          ││
│  │                                                          ││
│  │              IMAGE TRAITÉE (grand format)                 ││
│  │                                                          ││
│  │              + barre de couleur (colormap)               ││
│  │                                                          ││
│  └──────────────────────────────────────────────────────────┘│
│                                                              │
│  [💾 Télécharger]  ← Export PNG du résultat                  │
└──────────────────────────────────────────────────────────────┘
```

### Fonctionnement
1. Sélectionnez une méthode dans le menu déroulant (13 choix)
2. L'image est affichée en grand avec la **colormap** appropriée :
   - `gray` pour les lissages et Canny
   - `hot` pour Sobel+NMS, LoG, Contours Spline
   - `inferno` pour le spectre FFT
3. La **barre de couleur** (colorbar) indique l'intensité
4. Bouton **"💾 Télécharger"** pour sauvegarder en PNG

> [📷 CAPTURE 5A : **Vue Unique — Image originale** — Afficher l'onglet avec "🖼 Original" sélectionné]

> [📷 CAPTURE 5B : **Vue Unique — CLAHE** — Contraste amélioré visible]

> [📷 CAPTURE 5C : **Vue Unique — Sobel + NMS** — Contours fins en colormap "hot"]

> [📷 CAPTURE 5D : **Vue Unique — Canny** — Contours nets avec les paramètres par défaut]

> [📷 CAPTURE 5E : **Vue Unique — FFT Spectre** — Spectre d'amplitude en colormap "inferno"]

> [📷 CAPTURE 5F : **Vue Unique — FFT Passe-haut** — Bords extraits par filtrage fréquentiel]

---

# ÉTAPE 6 — Onglet Vue Complète (Grille de Toutes les Méthodes)

➤ **Onglet :** 🖼 Vue complète  
➤ **Usage :** Visualiser toutes les 13 méthodes simultanément

## 6.1 Interface

```
┌──────────────────────────────────────────────────────────────┐
│  [🔍 Vue unique] [🖼 Vue complète] [⚖️ Comparaison] …       │
│                                                              │
│  Toutes les méthodes                                         │
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │🖼 Original│ │✨ CLAHE  │ │🔵 Gauss. │ │🟠 Bilat. │       │
│  │          │ │          │ │          │ │          │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │🟡 Médian │ │🟢 Spline │ │🔴 Sobel  │ │⚡ Canny  │       │
│  │          │ │  Lissée  │ │  + NMS   │ │          │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │🔮 LoG    │ │🌿 Cont.  │ │🟣 FFT    │ │📈 FFT HP │       │
│  │          │ │  Spline  │ │  Spectre │ │          │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│  ┌──────────┐                                               │
│  │📉 FFT BP │  (dernière ligne si nombre impair)           │
│  └──────────┘                                               │
│                                                              │
│  [💾 Télécharger la grille complète]                         │
└──────────────────────────────────────────────────────────────┘
```

### Fonctionnement
- Grille de **4 colonnes × 4 lignes**
- Chaque vignette affiche le nom de la méthode en titre
- La colormap est appliquée automatiquement selon le type de méthode
- Les vignettes non utilisées sont masquées
- Bouton **"💾 Télécharger"** pour exporter la grille complète en une seule image PNG

> [📷 CAPTURE 6A : **Vue Complète — Grille 4×4** — Toutes les 13 méthodes visibles simultanément]

> [📷 CAPTURE 6B : **Comparaison visuelle entre méthodes** — Zoom sur la grille pour voir les différences]

---

# ÉTAPE 7 — Onglet Comparaison (Côte à Côte)

➤ **Onglet :** ⚖️ Comparaison  
➤ **Usage :** Confronter deux méthodes avec carte de différence

## 7.1 Interface

```
┌──────────────────────────────────────────────────────────────┐
│  [🔍 Vue unique] [🖼 Vue complète] [⚖️ Comparaison] …       │
│                                                              │
│  ┌─────────────────────┐ ┌─────────────────────┐            │
│  │  Méthode A          │ │  Méthode B          │            │
│  │  [🖼 Original  ▼]   │ │  [⚡ Canny     ▼]   │            │
│  │                     │ │                     │            │
│  │                     │ │                     │            │
│  │   IMAGE A           │ │   IMAGE B           │            │
│  │                     │ │                     │            │
│  │                     │ │                     │            │
│  └─────────────────────┘ └─────────────────────┘            │
│                                                              │
│  Diff. moyenne : 0.3421  |  Max : 0.8912                    │
│                                                              │
│  [ ] Carte de différence                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  (si cochée)                                          │   │
│  │         CARTE DE DIFFÉRENCE |A−B|                    │   │
│  │         (colormap "plasma" — jaune = très différent) │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### Fonctionnement
1. Sélectionnez une **Méthode A** (gauche) et une **Méthode B** (droite)
2. Les deux résultats sont affichés côte à côte
3. Les statistiques de différence sont calculées :
   - **Diff. moyenne** : écart moyen entre les deux images
   - **Max** : écart maximal
4. Option **"Carte de différence"** : affiche `|A − B|` en colormap plasma
   - Zones **jaunes/vives** = fortes différences entre les méthodes
   - Zones **sombres** = méthodes produisant le même résultat

> [📷 CAPTURE 7A : **Comparaison Original vs Canny** — Deux images côte à côte]

> [📷 CAPTURE 7B : **Carte de différence activée** — Colormap plasma montrant les zones de divergence]

> [📷 CAPTURE 7C : **Comparaison Sobel+NMS vs LoG** — Comparaison de deux méthodes de détection]

---

# ÉTAPE 8 — Onglet Suppression Fond (Détourage)

➤ **Onglet :** ✂️ Suppression Fond  
➤ **Usage :** Détourer un sujet et le placer sur un nouveau fond

C'est l'onglet **le plus complet et le plus utilisé** de l'application.

## 8.1 Interface

```
┌──────────────────────────────────────────────────────────────┐
│  [🔍 Vue unique] … [✂️ Suppression Fond] …                  │
│                                                              │
│  ✂️ rembg (IA U²-Net)                                       │
│                                                              │
│  🔬 ÉTAPES DE TRAITEMENT                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │Saillance │ │  Masque  │ │  Masque  │ │  Alpha   │       │
│  │Spectrale │ │   Brut   │ │ Raffiné  │ │ Matting  │       │
│  │          │ │          │ │          │ │          │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│                                                              │
│  ─────────────────────────────────────────────────────────  │
│                                                              │
│  🎨 RÉSULTATS                                                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │   Original   │ │ Fond: #FFFFF │ │  Fond blanc  │        │
│  │              │ │              │ │              │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
│                                                              │
│  ─────────────────────────────────────────────────────────  │
│                                                              │
│  STATISTIQUES                                                │
│  ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐  │
│  │Pixels sujet│ │Pixels    │ │Couverture│ │Qualité alpha│  │
│  │  45 230    │ │fond 7 890│ │  85.1%   │ │    0.94     │  │
│  └────────────┘ └──────────┘ └──────────┘ └─────────────┘  │
│                                                              │
│  EXPORTS                                                     │
│  [💾 PNG transparent] [💾 Fond coloré] [💾 Fond blanc]       │
└──────────────────────────────────────────────────────────────┘
```

## 8.2 Les 4 Étapes de Diagnostic

L'onglet affiche **4 visualisations intermédiaires** qui montrent comment l'algorithme fonctionne :

### Étape 1 — Saillance Spectrale
Carte de chaleur (colormap inferno) montrant les zones que l'algorithme considère comme **visuellement importantes** (probablement le sujet).
- 🔥 Zones chaudes (jaune/blanc) = forte probabilité de premier plan
- ❄️ Zones froides (noir/violet) = probablement le fond

### Étape 2 — Masque Brut
Résultat direct de l'algorithme de segmentation avant raffinement.
- 🟢 Vert = premier plan (sujet)
- 🔴 Rouge = arrière-plan (fond)

### Étape 3 — Masque Raffiné
Masque après application du pipeline de raffinement (remplissage des trous, suppression des îlots, edge snapping, alpha matting). Plus propre, plus précis.

### Étape 4 — Alpha Matting
Carte en niveaux de gris montrant la **transparence** :
- ⚪ Blanc (1.0) = sujet opaque
- ⚫ Noir (0.0) = fond transparent
- 🔘 Gris = zone de transition douce (bords)

## 8.3 Les 3 Résultats Finaux

### Colonne 1 — Original
L'image source non modifiée, pour référence.

### Colonne 2 — Fond Coloré
Le sujet détouré placé sur le fond de la couleur choisie dans la barre latérale (par défaut : blanc `#ffffff`).

### Colonne 3 — Fond Blanc
Le sujet détouré sur fond blanc pur `#FFFFFF`.

## 8.4 Statistiques de Segmentation

| Métrique | Description |
|----------|-------------|
| **Pixels sujet** | Nombre de pixels classés comme premier plan |
| **Pixels fond** | Nombre de pixels classés comme arrière-plan |
| **Couverture** | Pourcentage de l'image occupé par le sujet |
| **Qualité alpha** | Score moyen d'alpha dans la zone sujet (proche de 1 = bonne segmentation) |

## 8.5 Exports

Trois boutons de téléchargement :
- **💾 PNG transparent** — Format RGBA avec canal alpha (fond transparent)
- **💾 Fond coloré** — Image avec le fond de couleur choisi
- **💾 Fond blanc** — Image avec fond blanc pur

> [📷 CAPTURE 8A : **Suppression Fond complète** — Les 4 diagnostics + 3 résultats + stats + exports visibles]

> [📷 CAPTURE 8B : **Diagnostic — Saillance Spectrale** — Carte de chaleur montrant les zones saillantes]

> [📷 CAPTURE 8C : **Diagnostic — Masque Brut vs Raffiné** — Comparaison avant/après raffinement]

> [📷 CAPTURE 8D : **Résultat — Fond transparent (PNG)** — Sujet détouré avec damier de transparence]

> [📷 CAPTURE 8E : **Résultat — Fond coloré** — Sujet sur fond bleu/rouge/vert personnalisé]

> [📷 CAPTURE 8F : **Message rembg non installé** — Bandeau d'information "pip install rembg onnxruntime"]

---

# ÉTAPE 9 — Paramètres Avancés de Suppression de Fond

➤ **Emplacement :** Barre latérale → Section "✂️ Suppression Arrière-plan"

## 9.1 Choix de la Méthode

```
┌──────────────────────────────────────────┐
│  ✂️ Suppression Arrière-plan            │
│                                          │
│  Méthode  [rembg (IA U²-Net)     ▼]     │
│           ┌──────────────────────┐       │
│           │ rembg (IA U²-Net)   │       │
│           │ GrabCut + Saillance │       │
│           │ Flood Fill Bordures │       │
│           │ K-means LAB         │       │
│           └──────────────────────┘       │
└──────────────────────────────────────────┘
```

## 9.2 Paramètres par Méthode

### Méthode 1 — rembg (IA U²-Net)

| Paramètre | Type | Valeurs | Description |
|-----------|------|---------|-------------|
| **Modèle IA** | Select | 16 modèles | Réseau de neurones à utiliser (u2net, isnet, birefnet, sam…) |
| **Post-traitement rembg** | Checkbox | ON/OFF | Post-traitement du masque par rembg |
| **Alpha Matting rembg** | Checkbox | ON/OFF | Active l'alpha matting interne de rembg |

Si Alpha Matting activé :
| Paramètre | Plage | Défaut |
|-----------|-------|--------|
| Seuil premier plan | 0 – 255 | 240 |
| Seuil arrière-plan | 0 – 255 | 10 |
| Érosion | 1 – 50 | 10 |

### Méthode 2 — GrabCut + Saillance

| Paramètre | Plage | Défaut | Description |
|-----------|-------|--------|-------------|
| **Itérations** | 3 – 15 | 10 | Nombre d'itérations GrabCut (plus = meilleur mais plus lent) |
| **Marge** | 1 – 80 | 15 | Largeur de la bordure forcée comme fond (en pixels) |

### Méthode 3 — Flood Fill Bordures

| Paramètre | Plage | Défaut | Description |
|-----------|-------|--------|-------------|
| **Tolérance** | 5 – 80 | 25 | Seuil de tolérance pour le remplissage (plus élevé = plus agressif) |

### Méthode 4 — K-means LAB

| Paramètre | Plage | Défaut | Description |
|-----------|-------|--------|-------------|
| **Clusters** | 2 – 5 | 3 | Nombre de groupes de couleur (k) |

> [📷 CAPTURE 9A : **Sélection de méthode** — Menu déroulant avec les 4 méthodes visibles]

> [📷 CAPTURE 9B : **Paramètres rembg** — Modèle IA, checkboxes Alpha Matting, curseurs seuils]

> [📷 CAPTURE 9C : **Paramètres GrabCut** — Curseurs itérations et marge]

> [📷 CAPTURE 9D : **Comparaison des 4 méthodes** — Même image traitée avec GrabCut, Flood Fill, K-means, et rembg]

---

# ÉTAPE 10 — Raffinement du Masque

➤ **Emplacement :** Barre latérale → Section "Raffinement du masque"

## 10.1 Options de Raffinement

```
┌──────────────────────────────────────────┐
│  Raffinement du masque                   │
│                                          │
│  [✓] Remplir les trous                 │
│  [✓] Supprimer îlots isolés            │
│  [✓] Edge snapping (Canny)             │
│  Rayon snap (px)  [──●──] 4            │
│  Alpha matting (px) [──●──] 7          │
└──────────────────────────────────────────┘
```

### Pipeline de raffinement (dans l'ordre)

| Étape | Option | Effet |
|-------|--------|-------|
| 1 | Fermeture morphologique | Referme les petites ouvertures dans le masque (noyau 5×5, 2 itérations) |
| 2 | **Remplir les trous** | Comble les régions de fond entourées de sujet (ex: entre les jambes) |
| 3 | **Supprimer îlots isolés** | Élimine les petites taches de bruit (< 0.3% de l'image) |
| 4 | **Edge snapping** | Aligne les bords du masque sur les contours Canny les plus proches |
| 5 | **Alpha matting** | Crée une transition douce sur `radius` pixels pour des bords naturels |

### Paramètres

| Paramètre | Plage | Défaut | Description |
|-----------|-------|--------|-------------|
| **Rayon snap** | 1 – 12 px | 4 | Distance max de recherche d'un contour Canny pour aligner le bord |
| **Alpha matting** | 0 – 20 px | 7 | Largeur de la zone de transition douce (0 = bords durs) |

> [📷 CAPTURE 10A : **Case "Remplir les trous" cochée vs décochée** — Comparaison avant/après fill holes]

> [📷 CAPTURE 10B : **Edge snapping activé** — Bords alignés sur les contours naturels de l'image]

> [📷 CAPTURE 10C : **Alpha matting = 0 vs 15** — Bords durs vs bords très doux]

> [📷 CAPTURE 10D : **Suppression îlots isolés** — Avant (taches parasites) / Après (masque propre)]

---

# ÉTAPE 11 — Fonctions de Post-Traitement

➤ **Emplacement :** Barre latérale → Sections "Couleur du fond", "🌟 Fonctions Avancées", "🎨 Retouches & Filtres"

## 11.1 Couleur du Fond

```
┌──────────────────────────────────────────┐
│  Couleur du fond                         │
│  [🎨 #ffffff]  ← Color picker           │
└──────────────────────────────────────────┘
```

Permet de choisir n'importe quelle couleur de fond via un **sélecteur de couleur** (color picker). Le sujet détouré sera placé sur ce fond.

## 11.2 Recadrage Automatique

```
[✓] Recadrage auto au sujet
```

Rogne l'image pour ne garder que la zone contenant le sujet + une marge de 20 pixels.

## 11.3 Reflet Miroir

```
[✓] Ajouter un reflet miroir
Opacité reflet  [──●──] 0.3
Longueur reflet [──●──] 0.3
```

Crée un reflet en miroir sous le sujet avec un **dégradé d'opacité** (disparaît progressivement).

## 11.4 Ajustements d'Image

```
┌──────────────────────────────────────────┐
│  🎨 Retouches & Filtres                 │
│                                          │
│  Filtre artistique [Aucun         ▼]    │
│                    Aucun                 │
│  Luminosité        Croquis (Sketch)      │
│  [-100 ──●── 100]  Bleu technique       │
│                    (Blueprint)           │
│  Contraste                               │
│  [-100 ──●── 100]                       │
│                                          │
│  Saturation                              │
│  [0.0 ──●── 2.0]                        │
│                                          │
│  Netteté                                 │
│  [0.0 ──●── 2.0]                        │
└──────────────────────────────────────────┘
```

| Ajustement | Plage | Défaut | Description |
|------------|-------|--------|-------------|
| **Luminosité** | -100 à 100 | 0 | Éclaircit ou assombrit l'image |
| **Contraste** | -100 à 100 | 0 | Augmente ou diminue le contraste |
| **Saturation** | 0.0 à 2.0 | 1.0 | 0 = N&B, 1 = normale, 2 = saturée |
| **Netteté** | 0.0 à 2.0 | 0.0 | Unsharp masking (0 = inchangé) |

> [📷 CAPTURE 11A : **Color picker** — Sélecteur de couleur avec fond bleu/rouge personnalisé]

> [📷 CAPTURE 11B : **Recadrage automatique** — Image rognée autour du sujet avec marge]

> [📷 CAPTURE 11C : **Reflet miroir** — Sujet avec reflet dégradé en dessous]

> [📷 CAPTURE 11D : **Ajustements Luminosité/Contraste** — Avant/Après réglages]

> [📷 CAPTURE 11E : **Saturation à 0** — Image en noir et blanc]

---

# ÉTAPE 12 — Incrustation Poisson (Photomontage)

➤ **Onglet :** 🎨 Composition & Poisson  
➤ **Prérequis :** Avoir chargé une **Image Cible (Fond)** dans la barre latérale

## 12.1 Principe

L'incrustation de Poisson (Poisson Image Editing, Pérez et al. 2003) est une technique mathématique qui permet de **fondre un objet dans un nouveau décor** de manière ultra-réaliste en résolvant l'équation de Poisson pour adapter les couleurs et les lumières.

## 12.2 Interface

```
┌──────────────────────────────────────────────────────────────┐
│  🐟 Incrustation Seamless (Poisson Blending)                 │
│                                                              │
│  Mode de fusion : [Normal  ▼]      Opacité : [──●──] 1.0    │
│                   Normal                                     │
│  TRANSFORMATION    Gradient Max                              │
│  Échelle  [──●──] 1.0                                        │
│  Rotation  [─●──] 0°                                         │
│  [ ] Harmoniser lumières                                     │
│                                                              │
│  OMBRE PORTÉE                                                │
│  [ ] Activer l'ombre                                         │
│  (si activé) Flou [──●──] 15   Opacité [──●──] 0.4          │
│              Décalage XY [──●──] 10                          │
│                                                              │
│  Position X  [─────●──] 0                                    │
│  Position Y  [─────●──] 0                                    │
│                                                              │
│  👁️ APERÇU DU POSITIONNEMENT                                │
│  ┌──────────────────────────────────────────────────────────┐│
│  │                                                          ││
│  │     Image cible (fond) + objet positionné en overlay     ││
│  │                                                          ││
│  └──────────────────────────────────────────────────────────┘│
│                                                              │
│  [🚀 Lancer l'Incrustation Poisson]                          │
│                                                              │
│  (résultat final après clic)                                 │
│  [💾 Télécharger Résultat]                                   │
└──────────────────────────────────────────────────────────────┘
```

## 12.3 Les 3 Modes de Fusion

Pour chaque pixel dans la zone à fusionner, l'algorithme résout l'équation de Poisson avec un champ de guidage différent :

| Mode | Champ de guidage | Effet visuel |
|------|------------------|--------------|
| **Normal** | Gradient de la source | Fusion classique — l'objet garde ses couleurs et textures |
| **Gradient Max** | max(Gradient source, Gradient fond) | Préserve les bords les plus forts des deux images — détails nets |
| **Gradient Min** | min(Gradient source, Gradient fond) | Effet "fantôme" — l'objet est plus transparent, fond visible à travers |

### Copier-Coller Simple (Alpha)
Mode sans équation de Poisson — simple superposition avec opacité réglable (0 à 1).

## 12.4 Options de Transformation

| Paramètre | Plage | Description |
|-----------|-------|-------------|
| **Échelle** | 0.05 – 5.0 | Redimensionne l'objet (1.0 = taille originale) |
| **Rotation** | -180° à 180° | Rotation en degrés |
| **Harmoniser lumières** | ON/OFF | Transfère les statistiques de couleur du fond vers l'objet (Reinhard 2001) |

## 12.5 Ombre Portée

Si activée :
- **Flou ombre** (0-50) : Adoucit les bords de l'ombre
- **Opacité ombre** (0.0-1.0) : Intensité de l'ombre
- **Décalage XY** (-50 à 50) : Direction et distance de l'ombre

## 12.6 Aperçu en Direct

Avant de lancer la fusion complète (coûteuse en calcul), un **aperçu en direct** montre :
- Le positionnement de l'objet sur le fond
- L'ombre portée (si activée)
- La transformation appliquée (échelle, rotation)

L'aperçu se met à jour instantanément quand vous modifiez les curseurs de position.

> ⚠️ Pour les grandes zones de fusion (> 800 000 pixels), un avertissement est affiché.

> [📷 CAPTURE 12A : **Interface Poisson complète** — Mode, transformation, ombre, position, aperçu]

> [📷 CAPTURE 12B : **Aperçu en direct** — Objet positionné sur le fond avant fusion]

> [📷 CAPTURE 12C : **Résultat final — Mode Normal** — Objet parfaitement fondu dans le décor]

> [📷 CAPTURE 12D : **Résultat final — Mode Gradient Max** — Détails nets préservés]

> [📷 CAPTURE 12E : **Avec ombre portée** — Objet avec ombre réaliste sur le sol]

> [📷 CAPTURE 12F : **Avec harmonisation des couleurs** — Teintes de l'objet adaptées au décor]

---

# ÉTAPE 13 — Traitement par Lot (Export ZIP)

➤ **Emplacement :** Barre latérale → Section "📦 Traitement par lot"  
➤ **Condition :** Avoir chargé **plusieurs images**

## 13.1 Interface

```
┌──────────────────────────────────────────┐
│  📦 Traitement par lot                   │
│                                          │
│  ┌──────────────────────────────────────┐│
│  │  Générer ZIP (fond supprimé)         ││
│  └──────────────────────────────────────┘│
│                                          │
│  (progression : "Traitement du lot…")    │
│                                          │
│  [💾 Télécharger le ZIP]                 │
└──────────────────────────────────────────┘
```

### Fonctionnement
1. Chargez **plusieurs images** dans le sélecteur de fichiers
2. Configurez la méthode de suppression de fond et les paramètres
3. Cliquez sur **"Générer ZIP (fond supprimé)"**
4. L'application traite toutes les images avec les mêmes paramètres
5. Un fichier ZIP est généré contenant toutes les images détourées au format PNG

> 📦 Le nom de chaque fichier dans le ZIP est : `nom_original_no_bg.png`

Si un watermark est configuré, il est appliqué à toutes les images du lot.

> [📷 CAPTURE 13A : **Bouton Générer ZIP** — Avant clic, avec 5 images chargées]

> [📷 CAPTURE 13B : **Spinner de traitement** — "Traitement du lot en cours..."]

> [📷 CAPTURE 13C : **Bouton Télécharger ZIP** — ZIP prêt à être téléchargé]

---

# ÉTAPE 14 — Onglet Avant/Après

➤ **Onglet :** 🌓 Avant/Après  
➤ **Usage :** Comparaison visuelle interactive avec curseur glissant

## 14.1 Interface

```
┌──────────────────────────────────────────────────────────────┐
│  [🔍 Vue unique] … [🌓 Avant/Après]                         │
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │                                                          ││
│  │         ◀═══════════ CURSEUR ═══════════▶                ││
│  │                                                          ││
│  │    ┌──────────────┐          ┌──────────────────────┐    ││
│  │    │              │          │                      │    ││
│  │    │   AVANT      │    ←→    │       APRÈS          │    ││
│  │    │   (Original) │          │   (Image traitée)    │    ││
│  │    │              │          │                      │    ││
│  │    └──────────────┘          └──────────────────────┘    ││
│  │                                                          ││
│  └──────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

### Fonctionnement
Un **curseur glissant** (slider) permet de révéler progressivement l'image traitée par-dessus l'image originale. En faisant glisser de gauche à droite, vous passez de l'image **Avant** (originale) à l'image **Après** (traitée).

C'est l'outil idéal pour **présenter** et **apprécier** le résultat d'un traitement.

> [📷 CAPTURE 14A : **Avant/Après — Curseur au milieu** — Moitié gauche = original, moitié droite = traité]

> [📷 CAPTURE 14B : **Avant/Après — Curseur à gauche** — Presque entièrement l'image traitée]

---

# ÉTAPE 15 — Filtres Artistiques et Effets Visuels

➤ **Emplacement :** Barre latérale → "🎨 Retouches & Filtres" → Filtre artistique

## 15.1 Les 3 Modes Artistiques

### Mode 1 — Aucun (par défaut)
Aucun filtre appliqué. L'image est affichée normalement.

### Mode 2 — Croquis (Sketch)
Transforme l'image en **dessin au crayon** :

```
Formule : Sketch = gray / (255 - blur(255 - gray))
```

1. L'image est convertie en niveaux de gris
2. Le négatif est flouté (Gaussien 21×21)
3. L'image grise est divisée par le complément du flou

**Résultat :** Effet crayonné avec des traits qui suivent les contours naturels.

### Mode 3 — Bleu Technique (Blueprint)
Transforme l'image en **plan d'architecte** :

1. Détection des contours Canny
2. Fond bleu profond `rgb(20, 60, 120)`
3. Contours en blanc pur
4. Grille superposée (lignes horizontales/verticales tous les 40px)

**Résultat :** Effet "plan technique" façon blueprint.

## 15.2 Combinaison avec d'autres effets

Les filtres artistiques sont appliqués **après** la suppression de fond. On peut donc avoir :
- Un sujet détouré en style croquis sur fond blanc
- Un sujet en blueprint sur fond transparent
- Un croquis avec watermark
- Etc.

> [📷 CAPTURE 15A : **Filtre Croquis** — Image transformée en dessin au crayon]

> [📷 CAPTURE 15B : **Filtre Blueprint** — Image en bleu technique avec grille]

> [📷 CAPTURE 15C : **Sujet détouré + Croquis** — Combinaison suppression fond + filtre sketch]

---

# ÉTAPE 16 — Filigrane (Watermark)

➤ **Emplacement :** Barre latérale → Section "🖋️ Filigrane (Watermark)"

## 16.1 Interface

```
┌──────────────────────────────────────────┐
│  🖋️ Filigrane (Watermark)               │
│                                          │
│  Texte  [© Mon Entreprise________]      │
│                                          │
│  Couleur   Opacité                       │
│  [🎨 #fff] [──●──] 0.6                   │
│                                          │
│  Taille texte  [──●──] 1.5               │
│  Position  [Bas-Droite          ▼]       │
│            Bas-Droite                    │
│  [✓] Contour de visibilité              │
│  [ ] Afficher sur toutes les vues        │
└──────────────────────────────────────────┘
```

## 16.2 Paramètres

| Paramètre | Valeurs | Défaut | Description |
|-----------|---------|--------|-------------|
| **Texte** | Chaîne libre | "" | Texte du filigrane (ex: "© Mon Nom") |
| **Couleur** | Color picker | #ffffff | Couleur du texte |
| **Opacité** | 0.0 – 1.0 | 0.6 | Transparence du filigrane |
| **Taille** | 0.5 – 5.0 | 1.5 | Échelle du texte |
| **Position** | 5 choix | Bas-Droite | Bas-Droite, Bas-Gauche, Haut-Gauche, Haut-Droite, Centre |
| **Contour** | ON/OFF | ON | Ajoute un contour pour la visibilité (noir si texte clair, blanc si texte sombre) |
| **Toutes les vues** | ON/OFF | OFF | Applique aussi le watermark dans les onglets Vue Unique/Complète/Comparaison |

## 16.3 Fonctionnement

- Utilise la police `FONT_HERSHEY_DUPLEX` d'OpenCV
- Double tracé : contour + texte principal
- Fusion par `cv2.addWeighted` pour l'opacité
- Marges de sécurité de 2% pour éviter le débordement
- Le contour s'adapte automatiquement (noir ou blanc) selon la luminosité du texte

> [📷 CAPTURE 16A : **Watermark activé** — "© CelloRide" en bas à droite, blanc avec contour]

> [📷 CAPTURE 16B : **Paramètres watermark** — Sélecteur de position montrant les 5 options]

> [📷 CAPTURE 16C : **Watermark sur fond transparent** — Filigrane visible sur PNG exporté]

---

# SYNTHÈSE — Tableau Récapitulatif des Fonctionnalités

## Parcours Utilisateur Complet

```
                       ┌──────────────────┐
                       │  LANCEMENT APP   │
                       │ streamlit run    │
                       │    code.py       │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ CHARGER IMAGE(S) │
                       │ (1 ou plusieurs) │
                       └────────┬─────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
          ▼                     ▼                     ▼
   ┌──────────────┐    ┌──────────────┐     ┌──────────────────┐
   │ ANALYSER LES │    │ SUPPRIMER LE │     │ FAIRE UN         │
   │  CONTOURS    │    │    FOND      │     │ PHOTOMONTAGE     │
   └──────┬───────┘    └──────┬───────┘     └────────┬─────────┘
          │                   │                      │
          ▼                   ▼                      ▼
   ┌──────────────┐    ┌──────────────┐     ┌──────────────────┐
   │ • Vue Unique │    │ • Choix algo │     │ • Charger fond   │
   │ • Vue Compl. │    │ • Paramètres │     │ • Transformer    │
   │ • Comparer   │    │ • Diagnostic │     │ • Positionner    │
   │ • Télécharger│    │ • Raffiner   │     │ • Mode fusion    │
   └──────────────┘    │ • Exporter   │     │ • Ombre portée   │
                       └──────┬───────┘     │ • Lancer fusion  │
                              │             │ • Télécharger    │
                              ▼             └──────────────────┘
                       ┌──────────────┐
                       │ POST-TRAITER │
                       │ • Watermark  │
                       │ • Recadrage  │
                       │ • Reflet     │
                       │ • Ajustements│
                       │ • Filtre art.│
                       └──────┬───────┘
                              │
                              ▼
                       ┌──────────────┐
                       │  EXPORTER    │
                       │ • PNG trans. │
                       │ • Fond coloré│
                       │ • Fond blanc │
                       │ • ZIP (lot)  │
                       └──────────────┘
```

## Tableau des Onglets et Leurs Fonctionnalités

| Onglet | Icône | Fonctionnalités principales |
|--------|-------|----------------------------|
| **Vue Unique** | 🔍 | Analyse d'une méthode à la fois, grand format, export individuel |
| **Vue Complète** | 🖼 | Grille 4×4 de toutes les méthodes, comparaison rapide, export global |
| **Comparaison** | ⚖️ | Côte-à-côte de 2 méthodes + carte de différence |A−B| |
| **Suppression Fond** | ✂️ | Détourage intelligent, 4 diagnostics, 3 résultats, stats, exports |
| **Composition & Poisson** | 🎨 | Photomontage seamless, transformation, ombre, modes de fusion |
| **Avant/Après** | 🌓 | Curseur glissant pour révéler progressivement le traitement |

## Tableau des Formats d'Export

| Export | Format | Contenu | Disponible dans |
|--------|--------|---------|-----------------|
| Vue Unique | PNG | Image traitée avec colormap | Onglet Vue Unique |
| Grille complète | PNG | Toutes les méthodes en grille 4×4 | Onglet Vue Complète |
| PNG transparent | PNG (RGBA) | Sujet détouré avec canal alpha | Onglet Suppression Fond |
| Fond coloré | PNG | Sujet sur fond de couleur au choix | Onglet Suppression Fond |
| Fond blanc | PNG | Sujet sur fond blanc pur | Onglet Suppression Fond |
| Résultat Poisson | PNG | Composition finale après fusion | Onglet Composition & Poisson |
| ZIP par lot | ZIP | Toutes les images détourées + watermark | Barre latérale |

## Fondements Mathématiques

| Concept | Application dans le projet |
|---------|---------------------------|
| **Convolution discrète 2D** | Sobel, Gaussien, LoG, Filtres |
| **Filtre Gaussien** | Lissage, pré-traitement anti-bruit |
| **CLAHE** | Amélioration du contraste local par égalisation d'histogramme adaptative |
| **Filtre Bilatéral** | Lissage avec préservation des bords |
| **Filtre Médian** | Suppression du bruit impulsionnel |
| **Splines Bicubiques** | Lissage par interpolation polynomiale |
| **NMS (Non-Maximum Suppression)** | Amincissement des contours à 1 pixel |
| **Algorithme de Canny** | Double seuillage avec hystérèse |
| **LoG (Laplacien du Gaussien)** | Dérivée seconde + zéro-croisements |
| **Transformée de Fourier 2D** | Analyse fréquentielle, spectre |
| **Filtre de Butterworth** | Passe-haut / Passe-bas dans le domaine fréquentiel |
| **Saillance Spectrale** (Hou & Zhang) | Détection des régions visuellement importantes |
| **GrabCut** (Rother et al.) | Segmentation par GMM + MRF + min-cut |
| **K-means** | Clustering dans l'espace CIELAB |
| **Flood Fill** | Remplissage par diffusion BFS/DFS |
| **Distance Transform** | Alpha matting pour bords doux |
| **Équation de Poisson** (Pérez et al.) | Fusion seamless par résolution d'EDP |
| **Transfert de couleur** (Reinhard et al.) | Harmonisation des teintes dans l'espace LAB |
| **Systèmes linéaires creux** | Résolution du Laplacien discret pour Poisson blending |

## Technologies utilisées

| Catégorie | Technologie | Version |
|-----------|------------|---------|
| **Framework web** | Streamlit | ≥1.30 |
| **Traitement d'image** | OpenCV (cv2) | ≥4.8 |
| **Calcul numérique** | NumPy | ≥1.24 |
| **Calcul scientifique** | SciPy | ≥1.11 |
| **Visualisation** | Matplotlib | ≥3.7 |
| **Manipulation d'images** | Pillow (PIL) | ≥10.0 |
| **IA (optionnel)** | rembg + onnxruntime | latest |
| **Langage** | Python | 3.10+ |

---

## 📸 GUIDE DES CAPTURES D'ÉCRAN REQUISES

Pour une présentation complète, **70 captures d'écran** sont recommandées, réparties comme suit :

| Étape | Captures | Pages à capturer |
|-------|----------|------------------|
| ÉTAPE 1 | 2 | Terminal lancement, navigateur à l'ouverture |
| ÉTAPE 2 | 2 | Interface complète, barre latérale |
| ÉTAPE 3 | 2 | Après chargement image, chargement multiple |
| ÉTAPE 4 | 3 | Sections paramètres (Lissage, Canny, FFT) |
| ÉTAPE 5 | 6 | Vue Unique : Original, CLAHE, Sobel+NMS, Canny, FFT Spectre, FFT HP |
| ÉTAPE 6 | 2 | Vue Complète : Grille 4×4, zoom |
| ÉTAPE 7 | 3 | Comparaison Original vs Canny, carte différence, Sobel vs LoG |
| ÉTAPE 8 | 6 | Suppression Fond : vue complète, saillance, masque brut/raffiné, transparent, fond coloré, info rembg |
| ÉTAPE 9 | 4 | Choix méthode, paramètres rembg, paramètres GrabCut, comparaison 4 méthodes |
| ÉTAPE 10 | 4 | Fill holes ON/OFF, edge snapping, alpha matting, remove islands |
| ÉTAPE 11 | 5 | Color picker, recadrage auto, reflet miroir, ajustements L/C, saturation N&B |
| ÉTAPE 12 | 6 | Interface Poisson, aperçu direct, résultat Normal, Gradient Max, ombre portée, harmonisation |
| ÉTAPE 13 | 3 | Bouton Générer ZIP, spinner, téléchargement |
| ÉTAPE 14 | 2 | Avant/Après curseur milieu, curseur à gauche |
| ÉTAPE 15 | 3 | Croquis, Blueprint, sujet détouré + sketch |
| ÉTAPE 16 | 3 | Watermark activé, positions, watermark sur transparent |
| **TOTAL** | **56** | |

---

*Document généré le 12 Juin 2026 — Présentation complète du parcours utilisateur de l'Application d'Analyse d'Image*  
*Prêt pour l'insertion de captures d'écran aux emplacements `[📷 CAPTURE: ...]`*
