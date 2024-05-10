import re
import matplotlib.pyplot as plt
import pandas as pd

# Function to read and parse data from a file
def parse_data_from_file(filename):
    # Pattern to capture both the existing and new data formats
    pattern = re.compile(r"(\w+(?:_\w+)+)_(\d+).*?avg intergen_time ([\d.]+)", re.DOTALL)
    results = []
    with open(filename, 'r') as file:
        data = file.read()
        for match in re.finditer(pattern, data):
            config_name = match.group(1)
            population_size = int(match.group(2))
            avg_intergen_time = float(match.group(3))
            results.append({
                "executable_type": config_name,
                "population_size": population_size,
                "avg_intergen_time": avg_intergen_time
            })
    return results

# Path to the file containing the data
file_path = 'last_result.txt'

# Parse the data
results = parse_data_from_file(file_path)

# Convert to DataFrame
df = pd.DataFrame(results)

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
for label, grp in df.groupby('executable_type'):
    grp.plot(ax=ax, kind='line', x='population_size', y='avg_intergen_time',
             label=label, marker='o', linestyle='-')

plt.title('Average Intergeneration Time by Population Size and Executable Type')
plt.xlabel('Population Size')
plt.ylabel('Average Intergeneration Time (ms)')
plt.legend(title="Executable Type", loc='best')
plt.grid(True)
plt.xscale('log')  # Use logarithmic scale for better visibility on x-axis
plt.show()
