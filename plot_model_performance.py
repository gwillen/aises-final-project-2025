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

# Define a color for each category/language
CATEGORY_COLORS = [
    {'category_name': "Overall", 'color': 'skyblue'},
    {'category_name': "C", 'color': 'lightcoral'},
    {'category_name': "Python", 'color': 'lightgreen'},
    {'category_name': "JS", 'color': 'gold'}
]

def plot_metric_bar_chart(performance_data, categories, category_colors_config, output_directory, metric_to_plot, chart_title):
    """
    Generates and saves a grouped bar chart for a single metric (confidence or correctness)
    showing model performance across different categories, grouped by model.

    Args:
        performance_data (list): List of dictionaries with model performance data.
        categories (list): List of strings for categories (e.g., "Overall", "C").
        category_colors_config (list): List of dictionaries defining a color for each category.
        output_directory (str): Directory to save the plot image.
        metric_to_plot (str): The metric to plot ("confidence" or "correctness").
        chart_title (str): The title for the chart.
    """
    num_models = len(performance_data)
    num_categories = len(categories)
    model_short_names = [data['model_short'] for data in performance_data]

    x_main_ticks = np.arange(num_models)  # Center positions for each model group
    total_bars_in_group = num_categories  # One bar per category within a model's group
    bar_width = 0.8 / total_bars_in_group # Width of a single bar

    fig, ax = plt.subplots(figsize=(14, 8)) # Adjusted figure size

    for cat_idx, category_name in enumerate(categories):
        # Find the color for the current category
        current_category_color_info = next((cc for cc in category_colors_config if cc['category_name'] == category_name), None)
        if not current_category_color_info:
            print(f"Warning: Color not found for category {category_name}. Using default.")
            category_color = 'gray'
        else:
            category_color = current_category_color_info['color']

        # Extract metric values for the current category across all models
        metric_values_for_category = [model_data[metric_to_plot][category_name] for model_data in performance_data]

        # Calculate offset for this category's bars across all model groups
        offset = (cat_idx - total_bars_in_group / 2) * bar_width + bar_width / 2
        bar_positions = x_main_ticks + offset

        ax.bar(bar_positions, metric_values_for_category, bar_width,
               label=category_name, color=category_color,
               edgecolor='black')

    # Set labels, title, and ticks
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(chart_title, fontsize=14)
    ax.set_xticks(x_main_ticks)
    ax.set_xticklabels(model_short_names, fontsize=10, rotation=15, ha="right")

    if metric_to_plot == "confidence":
        ax.set_ylim(90, 105)
        # Add a text annotation to indicate the Y-axis break/start point
        # Position it relative to the axes (0,0 is bottom-left, 1,1 is top-right of axes)
        ax.text(0.01, 0.01, 'Note: Y-axis starts at 90%',
                transform=ax.transAxes, fontsize=8, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
    else:
        ax.set_ylim(0, 105)

    ax.legend(title='Category/Language', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout for legend

    filename_metric_part = metric_to_plot.lower().replace(' ', '_')
    plot_filename = os.path.join(output_directory, f"model_{filename_metric_part}_barchart_by_model.png")
    plt.savefig(plot_filename)
    print(f"Saved bar chart: {plot_filename}")
    plt.close(fig)

def main():
    """
    Main function to generate the bar charts for confidence and correctness, grouped by model.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

    plot_metric_bar_chart(MODEL_PERFORMANCE_DATA, CATEGORIES, CATEGORY_COLORS, OUTPUT_DIR,
                          metric_to_plot="confidence",
                          chart_title="Model Confidence")

    plot_metric_bar_chart(MODEL_PERFORMANCE_DATA, CATEGORIES, CATEGORY_COLORS, OUTPUT_DIR,
                          metric_to_plot="correctness",
                          chart_title="Model Correctness")

    print("\nAll bar charts generated successfully.")

if __name__ == "__main__":
    main()
