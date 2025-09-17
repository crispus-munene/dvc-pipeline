# **Reproducible Data and ML Pipeline using DVC**

DVC is a tool used for versioning, tracking, and automating ML projects.
It extends Git to handle things that Git alone struggles with in ML, including:

   - Large data files: Git does not handle large data files well. DVC stores these outside of Git, eg S3, local cache
   - Reproducible pipelines: Each ml workflow stage has dependencies and outputs. DVC ensures if nothing changes, a stage won't rerun.
   - Experiment management: Enables you to run experiments with different hyperparams, for example:

   ```bash
   dvc exp run -S model.c=10
   ```

This project is an end to end ml pipeline using dvc to handle large datasets, creating reproducible pipelines, in that even months after, it stil runs the same way and produces same outputs, as a stage won't rerun if nothing changes.

### Quick glance at the model results

| Model                  | Recall score   |
|------------------------|----------------|
| Support Vector Machine | 0.928          |

**Production model used**: Support Vector Machine
**Metrics used**: Recall

## Running the app locally
1. Initialize git repository

    ```bash
    git init
    ```

2. Clone the project

    ```bash
    git clone https://github.com/crispus-munene/dvc-pipeline.git
    ```

3. Enter the project directory

    ```bash
    cd dvc-pipeline
    ```

4. Create and activate a virtual environment (choose one)

   ```bash
   python -m venv .venv
   source .venv/bin/activate

5. Install dependencies using uv
    ```bash
    uv pip install -r pyproject.toml
    ```

6. Running the ml pipeline:
    ```bash
    dvc repro
    ```

8. Run the mlflow ui
    ```bash
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    ```
9. View metrics
    ```bash
    dvc metrics show
    ```

10. View plots
    ```bash
    dvc plots show
    ```
   Copy the index.html to the browser