from utils import main, process_data, dummy_data
import numpy as np

data = dummy_data(n = 785+794+401+304+213+120+1, seed = 1234, path = "gas.csv")
#sources = np.unique(data["source"])
data.loc[data['source'] == 6, 'age'] = np.nan
data.loc[data['source'] == 9, 'sex'] = np.nan
data.loc[data["source"] == 6, 'comorb_cancer'] = 1
data.loc[0:785, "source"] = 0
data.loc[785:785+794, "source"] = 1
data.loc[785+794:785+794+401, "source"] = 2
data.loc[785+794+401:785+794+401+304, "source"] = 3
data.loc[785+794+401+304:785+794+401+304+213, "source"] = 4
data.loc[785+794+401+304+213:785+794+401+304+213+120, "source"] = 5
data = process_data(data, "brutalize_impute")

alpha = 0.05
results = main(data,
                alpha)

p_val = results['p_val']
clustering = results['clustering']
print(results)
print(f"Transportability check p-value: {p_val}")
print(f"Obtained clusters: {clustering}")