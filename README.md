# Sedimentary MCMC

A Python toolkit for analyzing sedimentary sequences using Markov Chain Monte Carlo (MCMC) methods, wavelet transforms, and dynamic time warping.

## Overview

This package provides tools for:
- **Log blocking** using continuous wavelet transform (CWT) to identify natural boundaries in stratigraphic signals
- **Facies clustering** using Self-Organizing Maps (SOM) with 2D colormap visualization
- **Transition probability matrix** construction from observed facies sequences
- **Synthetic log generation** via Markov chains
- **Log correlation** using Dynamic Time Warping (DTW)
- **Monte Carlo simulation** to assess correlation significance

## Installation

```bash
pip install -r requirements.txt
```

## Dependencies

- numpy
- matplotlib
- scipy
- librosa
- scikit-image
- minisom
- tqdm

## Usage

See the Jupyter notebook `MCMC_for_laminae.ipynb` for a complete workflow example.

### Quick Start

```python
import sedimentary_mcmc as sedmc
import matplotlib.pyplot as plt

# Load and convert image to log
image = plt.imread('data/wheeler_diagram.png')
log = sedmc.image_to_log(image, channel=0, invert=True)

# Block the log using CWT
md, blocked_log, bounds, props = sedmc.log_blocking(log, scale=1)

# Cluster facies using SOM
som_shape = (3, 3)
cluster_index, som, data_normalized = sedmc.cluster_facies_data(
    vsh, ths, som_shape=som_shape, init_method='pca'
)

# Generate 2D colormap for visualization
colormap, colormap_2d = sedmc.generate_2d_colormap(som_shape)

# Build transition matrix
M = sedmc.transition_matrix(cluster_index[::-1])

# Generate synthetic facies sequence
vsh_chain, ths_chain = sedmc.facies_chain(M, vsh, ths, cluster_index, chain_length=370)

# Correlate logs using DTW
p, q, D = sedmc.correlate_logs(log1, log2, exponent=0.15)
```

## Example Data

The `data/` directory contains example core images from the Permian Basin:
- `peterson_wheeler_new.png` - Peterson core
- `core_5412_wheeler_new.png` - Core 5412
- `core_4431_wheeler_new.png` - Core 4431

These cores consist of laminae that seem to correlate between cores that are located hundreds of meters or even kilometers apart.

## License

MIT License

## Author

Zoltan Sylvester
