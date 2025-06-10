import pandas as pd
import numpy as np
from scipy.stats import chi2
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.cluster.hierarchy import linkage, fcluster

def get_vectors(params, num_vectors):
    temp = params.to_numpy()
    vectors = np.array([temp[j::num_vectors] for j in range(num_vectors)])
    return vectors

def get_linear_gammas(params, num_vectors):
    full_vectors = get_vectors(params, num_vectors)
    index = full_vectors.shape[1] // 2
    
    return full_vectors[:, index:]

def load_data(path):
    data = pd.read_csv(path)
    data = data.dropna()

    data.rename(columns={'30_day_mort': 'mortality_30_day'}, inplace=True)

    data["source"] = data["source"].astype("category")
    data["sex"]= data["sex"].astype("category")

    return data

def LRT(pooled_model, nested_model):
    LL_pooled = pooled_model.llf 
    LL_nested = nested_model.llf

    LR_stat = 2 * (LL_nested - LL_pooled)

    df_diff = nested_model.df_model - pooled_model.df_model

    return chi2.sf(LR_stat, df_diff)


def generate_nested_model(data):
    columns = data.columns
    features = " + ".join(columns[:-3])
    source = columns[-1]
    treatment = columns[-2]
    outcome = columns[-3]

    model = smf.glm(formula = f"{outcome} ~ C({source}) + C({source}):(({features}) * {treatment})",
                    data = data,
                    family=sm.families.Binomial()
                    ).fit()
    
    return model

def generate_pooled_model(data):
    columns = data.columns
    features = " + ".join(columns[:-3])
    treatment = columns[-2]
    outcome = columns[-3]

    model = smf.glm(formula = f"{outcome} ~ {features} * {treatment}",
                    data = data,
                    family=sm.families.Binomial()
                    ).fit()
    
    return model

def recursive_cluster(data,
                      num_features,
                      num_trials,
                      alpha = 0.05):
    if num_trials == 1:
        return np.ones((1))
    
    data["source"] = data["source"].astype("int")
    
    nested_model = generate_nested_model(data)
    pooled_model = generate_pooled_model(data)

    p_val = LRT(pooled_model, nested_model)
    
    if p_val > alpha:
        return np.ones((num_trials))
    else:
        if num_trials == 2:
            return np.array([1, 2])
        
        vectors = get_linear_gammas(nested_model.params, num_trials)

        Z = linkage(vectors, method='complete') 
                
        dists = Z[:, 2]
        diffs = np.diff(dists)
        
        index = np.argmax(diffs)
        t = (dists[index] + dists[index + 1]) / 2
        
        clusters = fcluster(Z, t=t, criterion='distance') 
        final_clusters = np.zeros_like(clusters)
        
        cluster_counter = 0

        for i in np.unique(clusters):
            trials = [trial for trial, cluster in enumerate(clusters) if cluster == i]
            curr_data = data[data['source'].isin(trials)].copy()
            
            value_map = {val: idx for idx, val in enumerate(trials)}
            curr_data['source'] = curr_data['source'].map(value_map)
            
            new_clusters = recursive_cluster(curr_data,
                                             num_features,
                                             len(trials),
                                             alpha)
            
            new_clusters += cluster_counter
            cluster_counter += int(np.unique(new_clusters).shape[0])
            for idx, trial in enumerate(trials):
                final_clusters[trial] = new_clusters[idx]

    return final_clusters  

def main(data,
         alpha = 0.05):
    
    num_features = len(data.columns) - 3
    num_trials = data["source"].nunique()

    nested_model = generate_nested_model(data)
    pooled_model = generate_pooled_model(data)

    p_val = LRT(pooled_model, nested_model)

    results = {}

    results["p_val"] = p_val
    results["clustering"] = recursive_cluster(data = data,
                             num_features = num_features,
                             num_trials = num_trials,
                             alpha = alpha)
    
    return results

def dummy_data(n = 1000, 
               seed = 42, 
               path = None):

    data = {
        'sex': np.random.binomial(1, 0.5, n),
        'age': np.random.normal(65, 15, n),
        'rr': np.random.normal(18, 4, n),
        'dbp': np.random.normal(80, 10, n),
        'sbp': np.random.normal(120, 15, n),
        'temp': np.random.normal(36.8, 0.5, n),
        'hr': np.random.normal(75, 12, n),
        'spo2': np.random.normal(96, 2, n),
        'creat': np.random.normal(1.0, 0.3, n),
        'sodium': np.random.normal(138, 3, n),
        'urea': np.random.normal(5, 1.5, n),
        'crp': np.random.normal(10, 5, n),
        'glucose': np.random.normal(100, 20, n),
        'wbc': np.random.normal(7, 2, n),
        'comorb_cancer': np.random.normal(0.2, 0.4, n),
        'comorb_liver': np.random.normal(0.1, 0.3, n),
        'comorb_chf': np.random.normal(0.3, 0.4, n),
        'comorb_renal': np.random.normal(0.25, 0.35, n),
        'comorb_diabetes': np.random.normal(0.4, 0.45, n),
        'comorb_copd': np.random.normal(0.2, 0.4, n),
        '30_day_mort': np.random.binomial(1, 0.15, n),
        'treatment': np.random.binomial(1, 0.5, n),
        'source': np.random.choice([0, 1, 6, 9], size=n)
    }

    df = pd.DataFrame(data)

    missing_rate = 0.05
    for col in df.columns:
        missing_indices = np.random.choice(df.index, size=int(n * missing_rate), replace=False)
        df.loc[missing_indices, col] = np.nan

    if path is not None:
        df.to_csv(path, index=False)  
    
    return df
