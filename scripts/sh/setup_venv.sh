#!/usr/bin/env bash
# setup_venv.sh
# Creates a Python virtual environment, installs requirements, and registers a Jupyter kernel for notebooks.
# Intended for Linux systems.

set -euo pipefail
IFS=$'\n\t'

# Configurable variables
VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"
KERNEL_NAME="skin-moles-ai"
PYTHON_CMD="python3"

print_help() {
  cat <<EOF
Usage: $0 [--venv DIR] [--python PYTHON] [--req FILE] [--kernel NAME]

Options:
  --venv DIR      Virtualenv directory to create (default: .venv)
  --python PYTHON Python executable to use (default: python3)
  --req FILE      Requirements file path (default: requirements.txt)
  --kernel NAME   Jupyter kernel name to register (default: skin-moles-ai)
  -h, --help      Show this help message
EOF
}

# Parse args
while [[ ${#} -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_DIR="$2"; shift 2;;
    --python)
      PYTHON_CMD="$2"; shift 2;;
    --req)
      REQUIREMENTS_FILE="$2"; shift 2;;
    --kernel)
      KERNEL_NAME="$2"; shift 2;;
    -h|--help)
      print_help; exit 0;;
    *)
      echo "Unknown argument: $1" >&2; print_help; exit 2;;
  esac
done

echo "Using python: ${PYTHON_CMD}"

# Check python availability
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
  echo "${PYTHON_CMD} not found. Trying 'python'..." >&2
  if command -v python >/dev/null 2>&1; then
    PYTHON_CMD=python
  else
    echo "No suitable Python executable found. Install Python 3 and retry." >&2
    exit 3
  fi
fi

# Create venv
if [[ -d "$VENV_DIR" ]]; then
  echo "Virtualenv directory '$VENV_DIR' already exists. Skipping creation." 
else
  echo "Creating virtual environment in '$VENV_DIR'..."
  "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

# Activate venv (in-script)
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Upgrade pip and install build tools
echo "Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel

# Install requirements if file exists
if [[ -f "$REQUIREMENTS_FILE" ]]; then
  echo "Installing packages from $REQUIREMENTS_FILE..."
  python -m pip install -r "$REQUIREMENTS_FILE"
else
  echo "Requirements file '$REQUIREMENTS_FILE' not found. Skipping 'pip install -r'." 
fi

# Ensure jupyter and ipykernel installed for notebooks
echo "Installing jupyter and ipykernel (if not already present)..."
python -m pip install jupyter ipykernel

# Register kernel
echo "Registering Jupyter kernel named '$KERNEL_NAME'..."
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Python (${KERNEL_NAME})"

echo "Setup complete. To activate the virtualenv run:\n  source ${VENV_DIR}/bin/activate"
