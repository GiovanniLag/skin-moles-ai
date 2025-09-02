import yaml

def read_yaml(fname: str) -> dict:
    with open(fname, 'r') as f:
        return yaml.safe_load(f)

def write_yaml(fname: str, data: dict) -> None:
    with open(fname, 'w') as f:
        yaml.safe_dump(data, f)
