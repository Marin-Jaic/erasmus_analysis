from src.utils import main, load_data, process_data
import argparse
import pickle
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(
    description = "Real data experiment"
)

parser.add_argument('--input', type=str, help='Path to .csv file')

args = parser.parse_args()
path = args.input

data_preprocess = ["drop", "brutalize_drop", "brutalize_impute", "impute"]

data = load_data(path)
sorted_sources = data["source"].value_counts(ascending=True).index.tolist()

for approach in data_preprocess:
    approach_data = data.copy()
    
    for source in sorted_sources:
        processed_data = process_data(approach_data, approach)

        alpha = 0.05
        try:
            results = main(processed_data,
                        alpha)
        except Exception as e:
            print(f"Dropping source {source}")
            approach_data = approach_data[approach_data["source"] != source]
            continue

        p_val = results['p_val']
        clustering = results['clustering']

        print(f'''Data processed by {approach} method.
        Transportability check p-value: {p_val}
        Obtained clusters: {clustering}''')

        with open(f'erasmus_analysis_{approach}.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(results)
        break