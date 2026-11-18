# Tabimer — Dans le tableau

Navigation 3D à l'intérieur d'un tableau orientaliste, comme dans une scène de jeu vidéo figée.

Ouvrir `index.html` via un serveur local (`python3 -m http.server`) puis aller sur http://localhost:8000.

**Commandes** : ZQSD / WASD pour se déplacer, souris pour regarder, Espace / Maj pour monter / descendre,
molette pour avancer, R pour revenir au point de vue du peintre, T pour la visite guidée.
Sur mobile : joystick + glisser.

## Fonctionnement

La pièce est **modélisée** à partir de la perspective du tableau (méthode « tour into the picture ») :

- **Perspective mesurée sur la toile** : point de fuite (500, 380) au niveau des yeux des personnages debout,
  focale ≈ 930 px déduite de l'ellipse de la fontaine, œil à 1,55 m.
- **Géométrie** : une cour centrale bordée de deux portiques (colonnes à x ≈ ±2,7 m), sol, mur du fond à 13,6 m,
  alcôve à arcs (16,3 m) avec ses côtés et sa voûte, galerie à 10,5 m, plafonds, murs latéraux et 4ᵉ mur.
  La carte de profondeur (Depth Anything V2) est calibrée sur le sol et sert à classer chaque pixel.
- **Texture** : la peinture est projetée depuis l'œil du peintre sur cette géométrie. Les surfaces que le peintre
  ne montre pas (sol sous vos pieds, murs latéraux, 4ᵉ mur, plafond de la cour) reprennent ses motifs par repli miroir.
  Derrière les personnages, la pièce est « nettoyée » en prolongeant les motifs.
- **Personnages, colonnes, mobilier** : 29 découpes peintes, au contour exact au pixel près, qui pivotent
  pour vous faire face (comme des sprites). Depuis le point de vue du peintre, on retrouve exactement le tableau.

```
cd tools
python3 depth.py depth_anything_v2_vits.onnx ../assets/source.webp   # → disp.npy
python3 room.py                                                      # → ../assets/*
```
Le modèle ONNX : `depth_anything_v2_vits.onnx` (release v2.0.0 de fabio-sim/Depth-Anything-ONNX).
