from utils import main, load_data, dummy_data
import argparse
import pickle

parser = argparse.ArgumentParser(
    description = "Real data experiment"
)

parser.add_argument('--input', type=str, help='Path to .csv file')
args = parser.parse_args()

path = args.input

data = load_data(path)

alpha = 0.05
results = main(data,
                alpha)

p_val = results['p_val']
clustering = results['clustering']

print(f"Transportability check p-value: {p_val}")
print(f"Obtained clusters: {clustering}")

with open('erasmus_analysis.pkl', 'wb') as f:
    pickle.dump(results, f)
    