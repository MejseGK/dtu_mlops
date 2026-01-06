# m6_project

Project for M6 in S2 in MLOps course 02576

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).


## Run scripts

Run the scripts from the terminal like this (in order):

```bash
# 0. (Once) create & activate environment
conda create -n m6_env python=3.11
conda activate m6_env

# 1. Install dependencies and project
pip install -r requirements.txt # from ChatGPT
pip install -e .

# 2. Process the raw data
python src/m6_project/data.py data/raw data/processed

# 3. Train the model
python src/m6_project/train.py

# 4. Evaluate the trained model
python src/m6_project/evaluate.py --model-checkpoint model.pth

# 5. Visualize learned features (t-SNE)
python src/m6_project/visualize.py --model-checkpoint model.pth
```

Alternatively use invoke, e.g. in #3: invoke train

