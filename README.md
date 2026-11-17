# DATA3800-pollen-prediction
Machine learning project for forecasting pollen concentrations in Zurich using historical weather and pollen data.


## How to run the project

This project uses **Poetry** for dependency management.


### 1. Install dependencies

```bash
poetry install --no-root
```

### 2. To run the data pipelines

Run the birch pollen pipeline:

```bash
poetry run python src/oblig2_bjork_data_pipeline.py
```

Run the grass pollen pipeline:

```bash
poetry run python src/oblig2_gress_data_pipeline.py
```

### 3. To run the notebooks

The notebooks are located in the `notebooks/` folder.

Open the desired `.ipynb` file in VS Code, Jupyter Notebook or Google Colab.

If using VS Code, select the Poetry environment as the notebook kernel.




The dataset must be located at:

data/full_pollen_and_weather_data_Zurich.csv