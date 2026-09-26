## Repository structure

- `DataPrep` — preprocessing of the raw motion capture data: loading, gait cycle segmentation, and normalization before the data goes into the models.
- `Model` — the main Dense VAE models and their variants. `NEW_DATA_MAN-WOMAN.ipynb` and `NEW_DATA_WOMAN-MAN.ipynb` contain the main sex-based experiments (male vs female). The folder also contains the LSTM-VAE variant and other comparison/baseline notebooks used during development.
- `Generated_Model` — the generative model based on marker positions, used to generate new gait sequences and evaluate them.

Each folder contains additional notebooks from earlier experiments and alternative approaches tried during development, not only the notebooks listed above.

## Dataset

Riglet et al. (2024), 3D motion analysis dataset of healthy young adult volunteers walking and running on overground and treadmill. Scientific Data 11:556. https://doi.org/10.6084/m9.figshare.c.7056797
