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

data_drop_row = process_data(data, "brutalize_drop")

alpha_drop_row = 0.05
results_drop_row = main(data_drop_row,
                        alpha_drop_row)

p_val_drop_row = results_drop_row['p_val']
clustering_drop_row = results_drop_row['clustering']

print(f''' Data processed by dropping missing columns from the analysis and dropping rows with missing values.
Transportability check p-value: {p_val_drop_row}
Obtained clusters: {clustering_drop_row}''')

with open('erasmus_analysis_brutalize_drop.pkl', 'wb') as f:
    pickle.dump(results_drop_row, f)

data_drop_col= process_data(data, "brutalize_impute")

alpha_drop_col = 0.05
results_drop_col = main(data_drop_col,
                        alpha_drop_col)

p_val_drop_col = results_drop_col['p_val']
clustering_drop_col = results_drop_col['clustering']

print(f'''Data processed by dropping missing columns from the analysis and imputing the missing values.
Transportability check p-value: {p_val_drop_col}
Obtained clusters: {clustering_drop_col}''')

with open('erasmus_analysis_brutalize_impute.pkl', 'wb') as f:
    pickle.dump(results_drop_col, f)
    
data_imputed_col = process_data(data, "impute")

alpha_imputed_col = 0.05
results_imputed_col = main(data_imputed_col,
                        alpha_imputed_col)

p_val_imputed_col = results_imputed_col['p_val']
clustering_imputed_col = results_imputed_col['clustering']

print(f'''Data processed by imputing the missing columns.
Transportability check p-value: {p_val_imputed_col}
Obtained clusters: {clustering_imputed_col}''')

with open('erasmus_analysis_imputed_col.pkl', 'wb') as f:
    pickle.dump(results_imputed_col, f)