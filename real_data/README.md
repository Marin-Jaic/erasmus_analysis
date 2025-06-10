```markdown
# Erasmus Data Analysis Tool

## 🛠️ Installation

### Option 1: Using Conda (Recommended)
```bash
# Create and activate a new conda environment
conda create -n erasmus_env python=3.9
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

## 🚀 Usage
```bash
python main.py --input <path_to_data.csv>
```
Example:
```bash
python main.py --input data/erasmus_2023.csv
```

## 📁 Output
• `erasmus_analysis.pkl` - Analysis results file

## ⚠️ Troubleshooting
• Package conflicts? Run:
```bash
conda env remove -n erasmus_env
conda create -n erasmus_env python=3.9
```
• CSV requirements:
  - UTF-8 encoding
  - Required columns: ['sex', 'age', 'rr', 'dbp', 'sbp', 'temp', 'hr', 'spo2', 'creat', 'sodium', 'urea',
                        'crp', 'glucose', 'wbc', 'comorb_cancer', 'comorb_liver', 'comorb_chf', 'comorb_renal',
        'comorb_diabetes', 'comorb_copd', '30_day_mort', 'treatment', 'source']
  - No header row

## 📜 License
MIT License - See LICENSE file
```
