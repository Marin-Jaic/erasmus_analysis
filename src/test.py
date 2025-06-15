from utils import main, process_data, dummy_data
import numpy as np

data = dummy_data(n = 2618, seed = 1234, path = "gas.csv")
#sources = np.unique(data["source"])
data.loc[data['source'] == 6, 'age'] = np.nan
data.loc[data['source'] == 9, 'sex'] = np.nan

data = process_data(data, "brutalize_impute")

alpha = 0.05
results = main(data,
                alpha)

p_val = results['p_val']
clustering = results['clustering']

print(f"Transportability check p-value: {p_val}")
print(f"Obtained clusters: {clustering}")