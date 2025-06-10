
# Erasmus Data Analysis Tool

## Installation

### Option 1: Using Conda (Recommended)
```bash
conda create -n <env_name>
conda activate <env_name>
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

## Output
• `erasmus_analysis.pkl` - Analysis results file

