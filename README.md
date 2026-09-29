# Galactic Binaries Dataset Generator

This repository is the first component of a four-part project developed during an internship at [L2IT](https://www.l2it.in2p3.fr/):

1. [Dataset generation](https://github.com/TristanDonze/gb-dataset-gen)
2. [Source counting](https://github.com/TristanDonze/gb-sources-counting)
3. [Source separation](https://github.com/TristanDonze/gb-sources-separation)
4. [Parameter estimation](https://github.com/TristanDonze/gb-parameters-estimation)

This project generates synthetic datasets of Galactic Binary signals with [GBGPU](https://github.com/mikekatz04/GBGPU). The generated waveforms are intended for machine-learning tasks such as source counting, source separation, and parameter estimation.

## Requirements

- Python 3.12 or later
- A CUDA-compatible GPU and CUDA 12 environment
- [`uv`](https://docs.astral.sh/uv/)

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

```bash
git clone https://github.com/TristanDonze/gb-dataset-gen.git
cd gb-dataset-gen
uv sync
```

You can check that PyTorch, CUDA, and GBGPU are available with:

```bash
uv run python test/test_install_1.py
uv run python test/test_install_2.py
```

## Configuration

Generation settings are defined in [`config.py`](config.py). The main options are:

- output directory and dataset name;
- number of samples and block size;
- observation duration and sampling cadence;
- fixed parameters and Sobol sampling ranges;
- signal-overlap difficulty factor;
- optional SNR filtering and its accepted range.

Adapt these values before starting a generation. In particular, choose `dataset_block_size` according to the available GPU memory.

## Dataset generation

Run the generation pipeline with:

```bash
uv run python main.py
```

The dataset is generated in blocks and saved under the path configured by `scratch_dir`, `dataset_collection_name`, and `dataset_name`. The output contains:

- an HDF5 file with the waveforms, sampled parameters, and SNR values;
- a JSON file containing the generation settings, dataset structure, and summary statistics.
