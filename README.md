# Tabimer — Dans le tableau

Navigation 3D à l'intérieur d'un tableau orientaliste, comme dans une scène de jeu vidéo figée.

Ouvrir `index.html` via un serveur local (`python3 -m http.server`) puis aller sur http://localhost:8000.

**Commandes** : ZQSD / WASD pour se déplacer, souris pour regarder, Espace / Maj pour monter / descendre,
molette pour avancer, R pour revenir au point de vue du peintre, T pour la visite guidée.
Sur mobile : joystick + glisser.

## Fonctionnement

1. `tools/depth.py` : estimation de profondeur monoculaire (Depth Anything V2, ONNX) sur le tableau.
2. `tools/prep.py` : nettoyage de la profondeur (bords francs, lissage par objet, colonnes redressées et arrondies),
   création d'un calque d'arrière-plan sans personnages ni colonnes (motifs prolongés en miroir) pour combler
   ce qui se dévoile quand on se déplace, export de `assets/`.
3. `index.html` : Three.js reprojette chaque pixel le long du rayon de vue du peintre → deux maillages
   (premier plan déchiré aux discontinuités + arrière-plan continu), texturés avec la peinture d'origine.

```
cd tools && python3 depth.py depth_anything_v2_vits.onnx ../assets/source.webp   # → disp.npy
python3 prep.py
```
