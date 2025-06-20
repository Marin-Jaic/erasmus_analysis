import pandas as pd
import numpy as np
import sys

from typing import Literal

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import chi2

import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer

#TREATMENT NOT LISTED AS BINARY BECAUSE IT IS USED AS A NUMERIC VALUE IN THE MODEL
numeric_cols = [
        'age', 'rr', 'dbp', 'sbp', 'temp', 'hr', 'spo2',
        'creat', 'sodium', 'urea', 'crp', 'glucose', 'wbc', 'treatment'
    ]

categorical_cols = [
        'sex', 'comorb_cancer', 'comorb_liver', 'comorb_chf',
        'comorb_renal', 'comorb_diabetes', 'comorb_copd',
        'mortality_30_day', 'source'
    ]

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

    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns='Unnamed: 0')

    trials = np.unique(data["source"])
    new_labels = [i for i in range(len(trials))]

    for trial, label in zip(trials, new_labels):
        data.loc[data["source"] == trial, "source"] = label

    return data

def process_data(data,
                missing_values: Literal['impute', 'brutalize_drop', 'brutalize_impute', 'drop'] = "drop"):
    data.rename(columns={'30_day_mort': 'mortality_30_day'}, inplace=True)

    data[categorical_cols] = data[categorical_cols].astype("category")

    if missing_values == "impute":
        data = impute_missing(data)
        #data = drop_perfect_separators(data)
    elif missing_values == "brutalize_drop":
        data = brutalize(data, impute = False)
        #data = drop_perfect_separators(data)
    elif missing_values == "brutalize_impute":
        data = brutalize(data, impute = True)
        #data = drop_perfect_separators(data)
    else:
        data = data.dropna()
    
    data = drop_constant_cols(data)
    return data

def LRT(pooled_model, nested_model):
    LL_pooled = pooled_model.llf 
    LL_nested = nested_model.llf

    LR_stat = 2 * (LL_nested - LL_pooled)
    df_diff = nested_model.df_model - pooled_model.df_model

    return chi2.sf(LR_stat, df_diff)

def generate_nested_model(data, regularized = False):
    columns = data.columns
    features = " + ".join(columns[:-3])
    source = columns[-1]
    treatment = columns[-2]
    outcome = columns[-3]

    model = None
    if not regularized:
        model = smf.glm(formula = f"{outcome} ~ C({source}) + C({source}):(({features}) * {treatment})",
                        data = data,
                        family=sm.families.Binomial()
                        ).fit(maxiter = 1000)
    else:
        model = smf.glm(formula = f"{outcome} ~ C({source}) + C({source}):(({features}) * {treatment})",
                        data = data,
                        family=sm.families.Binomial()
                        ).fit_regularized(alpha = 0.1, L1_wt = 1)
    
    return model

def generate_pooled_model(data, regularized = False):
    columns = data.columns
    features = " + ".join(columns[:-3])
    treatment = columns[-2]
    outcome = columns[-3]

    model = None
    if not regularized:
        model = smf.glm(formula = f"{outcome} ~ ({features}) * {treatment}",
                        data = data,
                        family=sm.families.Binomial()
                        ).fit(maxiter = 1000)
    else:
        model = smf.glm(formula = f"{outcome} ~ ({features}) * {treatment}",
                        data = data,
                        family=sm.families.Binomial()
                        ).fit(alpha = 0.1, L1_wt = 1)
    
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
    
    results = {}

    num_features = len(data.columns) - 3
    num_trials = data["source"].nunique()

    results["sources"] = np.unique(data["source"])
    results["remaining_columns"] = data.columns
    results["data_per_source"] = data["source"].value_counts().to_dict()

    nested_model = generate_nested_model(data)
    pooled_model = generate_pooled_model(data)
    
    if np.isnan(nested_model.llf):
        raise ValueError("Log-likelihood is nan")
        
    
    temp = {}
    for src in data["source"].unique():
        subset = data[data["source"] == src]
        constant_cols = [col for col in data.columns if subset[col].nunique(dropna=False) <= 1 and col != "source" and col != "treatment" and col != "mortality_30_day"]
        temp[src] = constant_cols

    results["constant_cols"] = temp

    results["nested_llf"] = nested_model.llf
    results["pooled_llf"] = pooled_model.llf

    results["nested_df"] = nested_model.df_model
    results["pooled_df"] = pooled_model.df_model

    # results["nested_convergence"] = nested_model.mle_retvals
    # results["pooled_convergence"] = pooled_model.mle_retvals

    p_val = LRT(pooled_model, nested_model)

    results["p_val"] = p_val
    results["clustering"] = recursive_cluster(data = data,
                             num_features = num_features,
                             num_trials = num_trials,
                             alpha = alpha)
    
    return results

def drop_constant_cols(data):
    constant_cols = set()

    for src in data["source"].unique():
        subset = data[data["source"] == src]
        cols = [col for col in data.columns 
                if col != "source" and subset[col].nunique(dropna=False) <= 1]
        constant_cols.update(cols)

    if len(constant_cols) > 0:
        print(f"Dropping constant columns: {constant_cols}")
    return data.drop(columns=constant_cols)

def drop_perfect_separators(df, outcome_col="mortality_30_day", source_col="source"):
    df = df.copy()
    features = [col for col in df.columns if col not in [source_col, outcome_col]]
    separators = set()

    for src in df[source_col].unique():
        group = df[df[source_col] == src]
        
        for col in features:
            if col == "treatment":
                continue

            temp = group[[col, outcome_col]].dropna()

            if temp.empty:
                continue

            ctab = pd.crosstab(temp[col], temp[outcome_col])

            if (ctab == 0).any(axis=1).any():
                separators.add(col)

    print(f"Columns dropped due to perfect separation: {sorted(separators)}")
    return df.drop(columns=separators)

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
        'comorb_cancer': np.random.binomial(1, 0.5, n),
        'comorb_liver': np.random.binomial(1, 0.5, n),
        'comorb_chf': np.random.binomial(1, 0.5, n),
        'comorb_renal': np.random.binomial(1, 0.5, n),
        'comorb_diabetes': np.random.binomial(1, 0.5, n),
        'comorb_copd': np.random.binomial(1, 0.5, n),
        '30_day_mort': np.random.binomial(1, 0.15, n),
        'treatment': np.random.binomial(1, 0.5, n),
        'source': np.random.choice([0, 1, 2, 3], size=n)
        #'source': np.random.choice([0, 1, 3], size=n)
    }

    df = pd.DataFrame(data)

    missing_rate = 0.05
    for col in df.columns:
        if col in ['source', 'mortality_30_day', 'treatment']:
            continue 

        missing_indices = np.random.choice(df.index, size=int(n * missing_rate), replace=False)
        df.loc[missing_indices, col] = np.nan

    if path is not None:
        df.to_csv(path, index=False)  
    
    return df

def impute_missing(data):
    a = list(set(data.columns) & set(numeric_cols)) 
    b = list(set(data.columns) & set(categorical_cols)) 
    all_features = a + b
    sources = data['source'].unique()

    def train_and_impute(train_data, pred_data, col, predictors, is_binary):
        imp = SimpleImputer(strategy="mean")
        X_train_imp = imp.fit_transform(train_data[predictors])
        X_pred_imp = imp.transform(pred_data[predictors])

        y_train = train_data[col]
        if is_binary:
            y_train = y_train.round().astype(int)
            model = RandomForestClassifier(n_estimators=100)
        else:
            model = RandomForestRegressor(n_estimators=100)

        model.fit(X_train_imp, y_train)
        preds = model.predict(X_pred_imp)
        return preds

    for col in all_features:
        if col in ['source', 'mortality_30_day', 'treatment']:
            continue
        is_binary = col in b
        predictors = [c for c in all_features if c != col and c != "mortality_30_day" and c != "source" and c != "treatment" ]

        for src in sources:
            mask_src = data['source'] == src
            if data.loc[mask_src, col].isnull().all():
                train_data = data[(data['source'] != src) & data[col].notnull()]
                if train_data.empty:
                    continue
                pred_data = data[mask_src]
                preds = train_and_impute(train_data, pred_data, col, predictors, is_binary)
                data.loc[mask_src, col] = preds

    for col in all_features:
        if col in ['source', 'mortality_30_day', 'treatment']:
            continue
        is_binary = col in b
        predictors = [c for c in all_features if c != col and c != 'mortality_30_day' and c != "treatment" and c != "source"]

        for src in sources:
            mask_src = data['source'] == src
            mask_missing = mask_src & data[col].isnull()
            mask_present = mask_src & data[col].notnull()

            if not mask_missing.any() or not mask_present.any():
                continue 

            train_data = data[mask_present]
            pred_data = data[mask_missing]
            preds = train_and_impute(train_data, pred_data, col, predictors, is_binary)
            data.loc[mask_missing, col] = preds

    return data


def brutalize(df, impute = False):
    numeric_cols = [
        'age', 'rr', 'dbp', 'sbp', 'temp', 'hr', 'spo2',
        'creat', 'sodium', 'urea', 'crp', 'glucose', 'wbc', 'treatment'
    ]

    categorical_cols = [
            'sex', 'comorb_cancer', 'comorb_liver', 'comorb_chf',
            'comorb_renal', 'comorb_diabetes', 'comorb_copd',
            'mortality_30_day', 'source'
        ]

    df = df.copy() 

    source_col = "source"
    drop_cols = []

    for col in numeric_cols + categorical_cols:
        if col == 'source' or col == "treatment":
            continue
        for src in df[source_col].unique():
            if df.loc[df[source_col] == src, col].isnull().all():
                drop_cols.append(col)
                break  

    if drop_cols:
        print(f"Dropping columns missing entirely in one or more sources: {drop_cols}")
        df = df.drop(columns=drop_cols)
        numeric_cols = [col for col in numeric_cols if col not in drop_cols]
        categorical_cols = [col for col in categorical_cols if col not in drop_cols]

    if not impute:
        df = df.dropna()
        return df
    
    all_cols = numeric_cols + categorical_cols
    predictors_all = [col for col in all_cols if col != source_col and col != "treatment" and col != "mort_30_day"]

    for src in df[source_col].unique():
        df_src = df[df[source_col] == src]
        for col in all_cols:
            if col == source_col or col == "treatment" or col == "mortality_30_day":
                continue
            if df_src[col].isnull().sum() == 0:
                continue  # no missing values

            predictors = [p for p in predictors_all if p != col and p in df.columns]

            train_df = df_src[df_src[col].notnull()]
            test_df = df_src[df_src[col].isnull()]

            if train_df.empty or test_df.empty:
                continue

            
            imp = SimpleImputer(strategy="mean")
            X_train = imp.fit_transform(train_df[predictors])
            X_test = imp.transform(test_df[predictors])
            y_train = train_df[col]

            if col in categorical_cols:
                y_train = y_train.astype(int)
                model = RandomForestClassifier(n_estimators=100)
            else:
                model = RandomForestRegressor(n_estimators=100)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            df.loc[df[source_col] == src, col] = df.loc[df[source_col] == src, col].fillna(pd.Series(y_pred, index=test_df.index))

    return df