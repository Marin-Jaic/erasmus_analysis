from src.utils import main, load_data, process_data
import argparse
import pickle
import traceback

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
    results = {}

    results["starting_sizes"] = data["source"].value_counts(ascending=True).to_dict() 

    print(f"Running approach {approach}")
    for source in sorted_sources:
        processed_data = process_data(approach_data, approach)

        alpha = 0.05
        try:
            clustering_results = main(processed_data,
                        alpha)
            results["results"] = clustering_results

        except ValueError as e:
            print(f"Dropping source {source}")
            approach_data = approach_data[approach_data["source"] != source]
            continue
        except Exception as e:
            error = {}
            error["type"] = str(e)
            error["text"] = traceback.format_exc()

            results["error"] = error
            with open(f'erasmus_analysis_{approach}.pkl', 'wb') as f:
                pickle.dump(results, f)
            break
        
        with open(f'erasmus_analysis_{approach}.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        break
        
    with open(f'erasmus_analysis_{approach}.pkl', 'wb') as f:
            pickle.dump(results, f)