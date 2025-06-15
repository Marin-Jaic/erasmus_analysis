from src.utils import main, load_data, process_data
import argparse
import pickle

parser = argparse.ArgumentParser(
    description = "Real data experiment"
)

parser.add_argument('--input', type=str, help='Path to .csv file')
args = parser.parse_args()

path = args.input

data = load_data(path)

data_drop_col= process_data(data, "brutalize_impute")

alpha_drop_col = 0.05
results_drop_col = main(data,
                        alpha_drop_col)

p_val_drop_col = results_drop_col['p_val']
clustering_drop_col = results_drop_col['clustering']

print(f'''
    Data processed by dropping missing columns from the analysis.
    Transportability check p-value: {p_val_drop_col}
    Obtained clusters: {clustering_drop_col}
      ''')

with open('erasmus_analysis_drop_col.pkl', 'wb') as f:
    pickle.dump(results_drop_col, f)
    
data_imputed_col = process_data(data, "impute")

alpha_imputed_col = 0.05
results_imputed_col = main(data,
                        alpha_imputed_col)

p_val_imputed_col = results_imputed_col['p_val']
clustering_imputed_col = results_imputed_col['clustering']

print(f'''
    Data processed by dropping missing columns from the analysis.
    Transportability check p-value: {p_val_imputed_col}
    Obtained clusters: {clustering_imputed_col}
      ''')

with open('erasmus_analysis_imputed_col.pkl', 'wb') as f:
    pickle.dump(results_imputed_col, f)