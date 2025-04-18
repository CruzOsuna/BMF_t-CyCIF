import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numba import jit
import time

# Configuration
INPUT_CSV = 'shannon_results.csv'
OUTPUT_IMAGE = 'shannon_heatmap.png'
COLOR_MAP = 'viridis'
FIG_SIZE = (32, 24)
DPI = 600
X_LABEL_INTERVAL = 5
Y_LABEL_INTERVAL = 50
FONT_SCALE = 0.8
THRESHOLD = 1  # Minimum value considered as "high" for the Shannon index

def timeit(func):
    """Timing decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Executed {func.__name__} in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@jit(nopython=True)
def calculate_sorting_metric_numba(steps_row, threshold):
    """Numba-accelerated metric: first step exceeding the threshold."""
    for idx in range(len(steps_row)):
        if steps_row[idx] >= threshold:
            return idx
    return len(steps_row)  # Default if never exceeded

@timeit
def create_shannon_heatmap(csv_path):
    # Set style
    sns.set(font_scale=FONT_SCALE)
    
    # Load and process data
    df = pd.read_csv(csv_path)
    
    # Extract step columns as NumPy array for Numba
    step_columns = sorted(
        [col for col in df.columns if col.startswith('step_')],
        key=lambda x: int(x.split('_')[1])
    )
    steps_data = df[step_columns].values
    
    # Calculate sorting metrics using Numba
    sort_keys = np.empty(len(df))
    for i in range(len(df)):
        sort_keys[i] = calculate_sorting_metric_numba(steps_data[i], THRESHOLD)
    
    # Sort and clean up
    df['sort_key'] = sort_keys
    df = df.sort_values(by='sort_key', ascending=True).drop('sort_key', axis=1)
    
    # Generate labels
    step_numbers = [int(col.split('_')[1]) for col in step_columns]
    x_labels = [
        str(num) if (num-1) % X_LABEL_INTERVAL == 0 else "" 
        for num in step_numbers
    ]
    
    num_samples = df.shape[0]
    y_labels = [
        f"Point {i+1}" if i % Y_LABEL_INTERVAL == 0 else "" 
        for i in range(num_samples)
    ]

    # Create heatmap
    plt.figure(figsize=FIG_SIZE)
    ax = sns.heatmap(
        df[step_columns].values,
        cmap=COLOR_MAP,
        cbar_kws={'label': 'Shannon Index'},
        yticklabels=y_labels,
        xticklabels=x_labels
    )
    
    # Customize axes
    ax.set_xlabel('Expansion Step', labelpad=15)
    ax.set_ylabel('Sampling Points (Sorted)', labelpad=15)
    plt.title(f'Shannon Index Progression (Sorted by Speed to Reach ≥{THRESHOLD})', pad=25)
    
    # Rotate X-axis labels
    ax.set_xticklabels(
        ax.get_xticklabels(), 
        rotation=45, 
        ha='right',
        fontsize=10
    )
    
    # Add grid lines
    ax.hlines(
        y=range(0, num_samples, 5), 
        xmin=0, 
        xmax=len(step_columns),
        colors='white',
        linewidths=0.1
    )
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=DPI, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    start_time = time.perf_counter()
    create_shannon_heatmap(INPUT_CSV)
    total_time = time.perf_counter() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")