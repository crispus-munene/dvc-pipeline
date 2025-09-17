import yaml

def load_params(path="params.yaml"):
    """Load parameters from YAML file"""
    with open(path, "r") as f:
        return yaml.safe_load(f)