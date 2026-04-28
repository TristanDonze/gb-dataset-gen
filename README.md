# gb-dataset-gen

This project generates a synthetic dataset of Galactic Binaries (GBs) using the GBGPU library.

The dataset can be used for machine learning tasks such as cardinality estimation, parameter estimation, and other GB-related studies.

You can customize the generation process in `config.py`, including:

* Number of samples
* Parameter ranges
* Which parameters are fixed or randomly sampled
* Difficulty factor (`1 = hard`, `10 = medium`, `100 = easy`)
* Additional generation settings

It uses **uv** for Python dependency management and environment setup.


## Prerequisites

Install `uv` first if you do not already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
````

## Getting Started

Clone the repository and move into the project folder:

```bash
git clone https://github.com/TristanDonze/gb-dataset-gen.git
cd gb-dataset-gen
```

Install dependencies and create the virtual environment:

```bash
uv sync
```

This will install all packages listed in `pyproject.toml` and `uv.lock`.

Then ensure everything works properly by running the tests:

```bash 
uv run python test/test_install_1.py
uv run python test/test_install_2.py
```

## Running the Project

Run the main script with:

```bash
uv run python main.py
```