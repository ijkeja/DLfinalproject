# DLfinalproject

This repository is used for the final project for the Deep Learning in Physics course (2025) by Elias Münch, Franca Maat and Ivan Keja. We implement code from particle_transformer (https://github.com/jet-universe/particle_transformer) to try and reproduce some of the results from: Qu, Huilin, Congqiao Li, and Sitian Qian. "Particle transformer for jet tagging." International Conference on Machine Learning. PMLR, 2022.

It consists of three parts.

1. The folder JetClass contains all necessary code to train the Part model on the smaller JetClass dataset.
2. The folder TopLandscape contains all necessary code to train the ParticleNet and ParT models on the smaller Top dataset.
   - The file CreateDatasets_TopLandscape.ipynb creates the smaller Top datasets that we use for training of the models. The original dataset is the Top Quark Tagging Reference Dataset (https://zenodo.org/records/2603256).
   - The file TopLandscape_ParticleTransformer.ipynb contains everything needed to train the models ParT and ParT-f.t. on the smaller Top dataset as well as evaluating their performance.
   - The file TopLandscape_ParticleNet.ipynb contains everything needed to train the models ParticleNet and ParticleNet-f.t. on the smaller Top dataset as well as evaluating their performance.
3. The file Visualization_Datasets.ipynb contains the code used to obtain the particle-cloud representations plots of the jets in the Top and JetClass datasets.
