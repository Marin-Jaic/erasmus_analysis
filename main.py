from src.utils import main, load_data, process_data
import numpy as np
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

sources = list(np.unique(data["source"]))

for approach in data_preprocess:
    approach_data = data.copy()
    results = {}

    results["starting_sizes"] = None
    results["removal_order"] = []

    while len(results["removal_order"]) != len(sources):
        processed_data = process_data(approach_data, approach)

        if results["starting_sizes"] is None:
             results["starting_sizes"] = data["source"].value_counts(ascending=True).to_dict() 

        alpha = 0.05
        try:
            clustering_results = main(processed_data,
                        alpha)
            results["results"] = clustering_results

        except ValueError as e:
            source_min = processed_data["source"].value_counts(ascending=True).idxmin()
            print(f"Dropping source {source_min}")
            
            approach_data = approach_data[approach_data["source"] != source_min]
            results["removal_order"] = results["removal_order"] + [source_min]
            
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