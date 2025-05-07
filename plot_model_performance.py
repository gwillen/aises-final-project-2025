#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import numpy as np

# Data derived from the prompt
MODEL_PERFORMANCE_DATA = [
    {
        "model_full": "gpt-4.1-2025-04-14",
        "model_short": "GPT-4.1",
        "correctness": {"Overall": 69.12, "C": 64.32, "Python": 74.00, "JS": 68.38},
        "confidence": {"Overall": 98.75, "C": 98.86, "Python": 98.61, "JS": 98.83},
    },
    {
        "model_full": "claude-3-7-sonnet-20250219 (without thinking)**",
        "model_short": "Claude (no think)",
        "correctness": {"Overall": 57.94, "C": 56.10, "Python": 60.67, "JS": 56.79},
        "confidence": {"Overall": 96.22, "C": 95.51, "Python": 97.00, "JS": 96.10},
    },
    {
        "model_full": "claude-3-7-sonnet-20250219 (thinking)*",
        "model_short": "Claude (thinking)",
        "correctness": {"Overall": 93.60, "C": 95.15, "Python": 94.59, "JS": 86.11},
        "confidence": {"Overall": 98.48, "C": 98.32, "Python": 98.58, "JS": 98.59},
    },
    {
        "model_full": "gemma-3-27b-it",
        "model_short": "Gemma 3 27B",
        "correctness": {"Overall": 59.28, "C": 53.26, "Python": 65.00, "JS": 58.97},
        "confidence": {"Overall": 97.08, "C": 96.81, "Python": 97.73, "JS": 96.40},
    },
]

CATEGORIES = ["Overall", "C", "Python", "JS"]
OUTPUT_DIR = "output/calibration_graphs"

# Define color pairs (light for confidence, dark for correctness) for each model
MODEL_COLORS = [
    {'short_name': "GPT-4.1", 'confidence': 'lightskyblue', 'correctness': 'dodgerblue'},
    {'short_name': "Claude (no think)", 'confidence': 'palegreen', 'correctness': 'limegreen'},
    {'short_name': "Claude (thinking)", 'confidence': 'lightcoral', 'correctness': 'indianred'},
    {'short_name': "Gemma 3 27B", 'confidence': 'plum', 'correctness': 'mediumorchid'}
]

def plot_grouped_bar_chart(performance_data, categories, model_colors_config, output_directory):
    """
    Generates and saves a single grouped bar chart showing model confidence
    and correctness across different categories.

    Args:
        performance_data (list): List of dictionaries with model performance data.
        categories (list): List of strings for categories (e.g., "Overall", "C").
        model_colors_config (list): List of dictionaries defining colors for each model.
        output_directory (str): Directory to save the plot image.
    """
    num_categories = len(categories)
    num_models = len(performance_data)
    model_short_names = [data['model_short'] for data in performance_data]

    # Calculate bar positions
    x_main_ticks = np.arange(num_categories)  # Center positions for each category group
    total_bars_in_group = num_models * 2  # Each model has a confidence and a correctness bar
    bar_width = 0.8 / total_bars_in_group # Dynamically calculate bar width to fit (0.8 is group width)
                                         # Adjust 0.8 if more/less spacing is desired between groups

    fig, ax = plt.subplots(figsize=(16, 9)) # Adjusted figure size for better readability

    for i, model_data_entry in enumerate(performance_data):
        model_name = model_data_entry['model_short']

        # Find the color configuration for the current model
        # This assumes model_short_names from performance_data matches model_colors_config ordering
        # Or, more robustly, match by 'short_name' if order isn't guaranteed.
        # For this script, we assume the order in MODEL_PERFORMANCE_DATA and MODEL_COLORS matches.
        current_model_colors = model_colors_config[i]

        confidence_values = [model_data_entry['confidence'][cat] for cat in categories]
        correctness_values = [model_data_entry['correctness'][cat] for cat in categories]

        # Calculate offset for this model's bars within each group
        # The term `-(total_bars_in_group / 2)` centers the whole group of bars.
        # `+ 0.5 * bar_width` ensures the first bar starts at the edge of its allocated space.
        # `i * 2 * bar_width` shifts to the start of the i-th model's pair.
        base_offset_for_model_pair = (i * 2 - total_bars_in_group / 2) * bar_width

        conf_bar_positions = x_main_ticks + base_offset_for_model_pair + (0.5 * bar_width)
        corr_bar_positions = x_main_ticks + base_offset_for_model_pair + (1.5 * bar_width)

        ax.bar(conf_bar_positions, confidence_values, bar_width,
               label=f'{model_name} Confidence', color=current_model_colors['confidence'],
               edgecolor='grey') # Added edgecolor for better separation
        ax.bar(corr_bar_positions, correctness_values, bar_width,
               label=f'{model_name} Correctness', color=current_model_colors['correctness'],
               edgecolor='grey') # Added edgecolor

    # Set labels, title, and ticks
    ax.set_ylabel('Percentage (%)', fontsize=14)
    ax.set_title('Model Performance: Confidence vs. Actual Correctness by Category', fontsize=16)
    ax.set_xticks(x_main_ticks)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 105) # Ensure y-axis goes up to at least 100%

    ax.legend(title='Model & Metric', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout to make space for legend outside

    plot_filename = os.path.join(output_directory, "model_performance_barchart.png")
    plt.savefig(plot_filename)
    print(f"Saved grouped bar chart: {plot_filename}")
    plt.close(fig)

def main():
    """
    Main function to generate the grouped bar chart.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

    # Ensure MODEL_COLORS matches the order and models in MODEL_PERFORMANCE_DATA
    # For this script, we assume they align based on their definition order.
    # A more robust solution might involve matching by model_short_name if data sources could vary.

    plot_grouped_bar_chart(MODEL_PERFORMANCE_DATA, CATEGORIES, MODEL_COLORS, OUTPUT_DIR)

    print("\nGrouped bar chart generated successfully.")

if __name__ == "__main__":
    main()
