
# Erasmus Data Analysis Tool

## Installation

### Option 1: Using Conda (Recommended)
```bash
# Create and activate a new conda environment
conda create -n erasmus_env 
conda activate erasmus_env
conda install --file requirements.txt
```

### Option 2: Using pip
```bash
python -m venv erasmus_venv
source erasmus_venv/bin/activate  # Linux/Mac
.\erasmus_venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Usage
```bash
python main.py --input <path_to_data.csv>
```
Example:
```bash
python main.py --input data/erasmus_2023.csv
```

## Output
• `erasmus_analysis.pkl` - Analysis results file

