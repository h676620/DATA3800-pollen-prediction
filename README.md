# DATA3800-pollen-prediction
Machine learning project for forecasting pollen concentrations in Zurich using historical weather and pollen data.


## How to run the project

This project uses **Poetry** for dependency management.

### 1. Clone the repository

```bash
git clone <repository-url>
cd DATA3800-pollen-prediction
```

### 2. Install dependencies

```bash
poetry install --no-root
```

### 3. To run the data pipelines

Run the birch pollen pipeline:

```bash
poetry run python src/oblig2_bjork_data_pipeline.py
```

Run the grass pollen pipeline:

```bash
poetry run python src/oblig2_gress_data_pipeline.py
```

### 4. To run the notebooks

The notebooks are located in the `notebooks/` folder.

Open the desired `.ipynb` file in VS Code, Jupyter Notebook or Google Colab.

If using VS Code, select the Poetry environment as the notebook kernel.

