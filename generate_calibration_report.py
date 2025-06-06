#!/usr/bin/env python3
"""
Generates calibration reports from evaluation JSON files produced by evaluate_code_examples.py.

Creates a calibration graph for each confidence strategy found in the input file.
"""

import os
import json
import argparse
from scipy.stats import beta, norm
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional

def load_evaluation_data(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Loads evaluation data from a JSON file.

    Args:
        filepath: Path to the JSON evaluation file.

    Returns:
        The loaded data as a dictionary, or None if an error occurs.
    """
    if not os.path.exists(filepath):
        print(f"Error: Input file not found: {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "evaluations" not in data or "confidence_strategies" not in data:
            print(f"Error: Required keys ('evaluations', 'confidence_strategies') not found in {filepath}")
            return None
        return data
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file: {filepath}")
        return None
    except Exception as e:
        print(f"Error loading evaluation data from {filepath}: {e}")
        return None

def extract_confidence_data(
    evaluations: List[Dict[str, Any]],
    strategy: str,
    time_point: str
) -> Tuple[List[float], List[bool]]:
    """
    Extracts confidence scores and corresponding match outcomes for a specific strategy and time point.

    Args:
        evaluations: The list of evaluation results.
        strategy: The confidence strategy name (e.g., 'standard').
        time_point: 'before' or 'after'.

    Returns:
        A tuple containing two lists: confidence scores and match outcomes (boolean).
        Only includes data points where confidence was successfully extracted.
    """
    confidence_scores = []
    actual_outcomes = []
    confidence_key = f"{time_point}_confidence"

    for eval_item in evaluations:
        confidence_results = eval_item.get("confidence_results", {})
        if strategy in confidence_results:
            confidence = confidence_results[strategy].get(confidence_key)
            is_match = eval_item.get("match")

            # Only include if confidence is not None and match status is available
            if confidence is not None and is_match is not None:
                confidence_scores.append(confidence)
                actual_outcomes.append(is_match)

    from collections import Counter

    print(f"Extracted {len(confidence_scores)} data points for strategy '{strategy}' ({time_point})")
    confidence_scores_match = [c for c, o in zip(confidence_scores, actual_outcomes) if o]
    confidence_counts_match = Counter(confidence_scores_match)
    print(f"Confidence counts when match: {confidence_counts_match}")

    confidence_scores_not_match = [c for c, o in zip(confidence_scores, actual_outcomes) if not o]
    confidence_counts_not_match = Counter(confidence_scores_not_match)
    print(f"Confidence counts when not match: {confidence_counts_not_match}")

    return confidence_scores, actual_outcomes

def clopper_pearson(outcomes: List[bool], alpha: float = 0.05) -> List[float]:
    """
    Calculate error bars using the Clopper-Pearson confidence interval for a binomial proportion.
    https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval

    Args:
        count: Number of successes in the sample.
        outcomes: List of boolean outcomes (True for correct, False for incorrect).
        frac_correct: Fraction of successes in the sample.
        alpha: Confidence level (default: 0.05 for 95% confidence interval).

    Returns:
        A list containing the lower and upper bounds of the confidence interval.
    """
    # calculate error bars using clopper-pearson

    k = np.sum(outcomes)
    n = len(outcomes)
    frac_correct = k / n

    p_u, p_o = beta.ppf([alpha / 2, 1 - alpha / 2], [k, k + 1], [n - k + 1, n - k])
    if np.isnan(p_o):
        p_o = 1
    if np.isnan(p_u):
        p_u = 0
    return [np.abs(p_u-frac_correct), np.abs(p_o-frac_correct)]

def wilson_interval(outcomes: List[bool], alpha: float = 0.05) -> List[float]:
    """
    Calculate error bars using the Wilson interval for a binomial proportion.
    https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
    """
    k = np.sum(outcomes)
    n = len(outcomes)
    frac_correct = k / n

    z = norm.ppf(1 - alpha / 2)
    p_mid = (k + z**2 / 2) / (n + z**2)
    p_o = p_mid + (z / (n + z**2)) * np.sqrt((k * (n-k) / n) + (z**2 / 4))
    p_u = p_mid - (z / (n + z**2)) * np.sqrt((k * (n-k) / n) + (z**2 / 4))

    print(f"k: {k}, n: {n}, frac_correct: {frac_correct}")
    print(f"Wilson interval: {p_u}, {p_o}")
    print(f"Wilson interval error: {np.abs(p_u-frac_correct)}, {np.abs(p_o-frac_correct)}")

    return [np.abs(p_u-frac_correct), np.abs(p_o-frac_correct)]

def plot_calibration_graph(
    confidence_scores: List[float],
    actual_outcomes: List[bool],
    strategy_name: str,
    superforecast: bool,
    time_point: str,
    model_names: List[str],
    output_dir: str,
    num_bins: int = 10
) -> None:
    """
    Generates and saves a calibration graph.

    Args:
        confidence_scores: List of predicted confidence scores (0.0 to 1.0).
        actual_outcomes: List of actual outcomes (True for correct, False for incorrect).
        strategy_name: Name of the confidence strategy.
        time_point: 'before' or 'after'.
        model_names: List of model names.
        output_dir: Directory to save the plot.
        num_bins: Number of bins to divide the confidence scores into.
    """
    if not confidence_scores:
        print(f"Warning: No valid confidence data found for strategy '{strategy_name}' ({time_point}). Skipping plot.")
        return

    #superforecast_str = "(superforecast persona prompt)" if superforecast else "(no persona prompt)"

    scores_array = np.array(confidence_scores)
    outcomes_array = np.array(actual_outcomes)

    # Define bins (e.g., 0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 # For plotting

    # Assign each score to a bin
    bin_indices = np.digitize(scores_array, bin_edges[1:-1]) # Indices 0 to num_bins-1

    mean_confidences = []
    fraction_correct = []
    bin_counts = []
    error_bars = []
    for i in range(num_bins):
        in_bin = (bin_indices == i)
        count = np.sum(in_bin)
        bin_counts.append(count)

        if count > 0:
            mean_conf = np.mean(scores_array[in_bin])
            frac_correct = np.mean(outcomes_array[in_bin])

            mean_confidences.append(mean_conf)
            fraction_correct.append(frac_correct)

            #err5 = clopper_pearson(outcomes_array[in_bin], 0.05)
            #err10 = clopper_pearson(outcomes_array[in_bin], 0.1)
            #err20 = clopper_pearson(outcomes_array[in_bin], 0.2)
            #err5_wilson = wilson_interval(outcomes_array[in_bin], 0.05)
            #err10_wilson = wilson_interval(outcomes_array[in_bin], 0.1)
            err20_wilson = wilson_interval(outcomes_array[in_bin], 0.2)
            error_bars.append([err20_wilson])

            print(f"Bin {i}:")
            print(f"count: {count}")
            print(f"Error bars: {error_bars[-1]}")
            print(f"Mean confidence: {mean_conf}")
            print(f"Fraction correct: {frac_correct}")

        else:
            # Append values that allow plotting vs bin_centers if needed
             mean_confidences.append(bin_centers[i]) # Use bin center for x if empty
             fraction_correct.append(np.nan) # Use NaN for y if empty
             error_bars.append([[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]])

    # Filter out bins with no data for plotting the curve
    plot_mean_conf = [mc for i, mc in enumerate(bin_centers) if bin_counts[i] > 0]
    plot_frac_correct = [fc for i, fc in enumerate(fraction_correct) if bin_counts[i] > 0]
    plot_error_bars = [eb for i, eb in enumerate(error_bars) if bin_counts[i] > 0]

    # transpose error bars
    # currently: bin x width x [hi, low]
    print(f"plot_error_bars: {np.array(plot_error_bars).shape}")
    # want: width x [hi, low] x bin
    plot_error_bars = np.array(plot_error_bars).transpose(1, 2, 0).tolist()
    print(f"plot_error_bars: {np.array(plot_error_bars).shape}")
    #plot_error_bars = np.array(plot_error_bars).T.tolist()

    draw_error_bars = True
    draw_counts = True

    # make the font much larger:
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(8, 8))
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], 'k:', label='Perfect Calibration')
    # Plot calibration curve
    if plot_mean_conf: # Only plot if there's data
        plt.plot(plot_mean_conf, plot_frac_correct, 'o-', label=f'{strategy_name.capitalize()} ({time_point.capitalize()})', markersize=8)
        if draw_error_bars:
            plt.errorbar(plot_mean_conf, plot_frac_correct, yerr=plot_error_bars[0], fmt='none', ecolor='black', capsize=5)
            #plt.errorbar(plot_mean_conf, plot_frac_correct, yerr=plot_error_bars[1], fmt='none', ecolor='red', capsize=5)
            #plt.errorbar(plot_mean_conf, plot_frac_correct, yerr=plot_error_bars[2], fmt='none', ecolor='blue', capsize=5)
    # Add counts to the plot (optional, can be noisy)
    if draw_counts:
        for i in range(num_bins):
            if bin_counts[i] > 0 and not np.isnan(fraction_correct[i]):
             plt.text(mean_confidences[i], fraction_correct[i] + 0.02, f'n={bin_counts[i]}', ha='center', va='bottom', fontsize=8)

    assert len(model_names) == 1, f"Expected exactly one model name, got {len(model_names)}"
    model_name = list(model_names)[0]

    plt.xlabel("Mean Predicted Confidence in Bin")
    plt.ylabel("Fraction of Positives (Actual Accuracy in Bin)")
    plt.title(f"{model_name} \
({strategy_name} / {time_point})")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    plt.gca().set_aspect('equal', adjustable='box') # Ensure square plot with equal axes

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{strategy_name}_{time_point}.png"
    filepath = os.path.join(output_dir, filename)
    try:
        plt.savefig(filepath)
        print(f"Saved calibration plot to: {filepath}")
    except Exception as e:
        print(f"Error saving plot to {filepath}: {e}")
    plt.close() # Close the figure to free memory

# XXX
time_points = ['before'] # XXX , 'after']

def main():
    """Main function to parse arguments and generate reports."""
    parser = argparse.ArgumentParser(description='Generate calibration graphs from evaluation data.')
    parser.add_argument('--input-files', required=True,
                        help='Path to the batch evaluation JSON files, separated by commas.')
    parser.add_argument('--output-dir', default='output/reports',
                        help='Base directory to save the generated report folders (default: output/reports).')
    parser.add_argument('--bins', type=int, default=20,
                        help='Number of bins for calibration plots (default: 20 for 5% bins).')

    args = parser.parse_args()

    evaluations = []
    strategies = set()
    model_names = set()
    input_filename_bases = []

    input_files = args.input_files.split(',')
    for input_file in input_files:
        print(f"Loading evaluation data from: {input_file}")
        data = load_evaluation_data(input_file)
        if not data:
            return
        evaluations.extend(data.get("evaluations", []))
        strategies.update(data.get("confidence_strategies", []))
        model_names.add(data.get("model", "unknown_model"))
        input_filename_bases.append(os.path.splitext(os.path.basename(input_file))[0])

    output_dirname = "___".join(input_filename_bases)
    # Create a run-specific output directory
    run_output_dir = os.path.join(args.output_dir, output_dirname)
    try:
        os.makedirs(run_output_dir, exist_ok=True)
        print(f"Ensured output directory for this run: {run_output_dir}")
    except OSError as e:
        print(f"Error creating output directory {run_output_dir}: {e}")
        return

    if not evaluations:
        print("No evaluations found in the input file.")
        return
    if not strategies:
        print("Warning: No confidence strategies listed in the input file.")
        # Attempt to infer strategies if missing, though this shouldn't happen with current evaluate script
        strategies = list(evaluations[0].get("confidence_results", {}).keys()) if evaluations else []
        if not strategies:
            print("Could not determine strategies. Exiting.")
            return
        print(f"Inferred strategies: {', '.join(strategies)}")

    print(f"Strategies: {strategies}")
    strategies_to_plot = ["standard", "inverse", "onetoten", "onetoten_decimal"]
    # Filter strategies to only include standard and inverse for now
    target_strategies = [s for s in strategies if s in strategies_to_plot]

    if not target_strategies:
        print("Neither 'standard' nor 'inverse' strategies found in the data. No plots to generate.")
        return

    print(f"Found {len(evaluations)} evaluations.")
    print(f"Processing strategies for calibration plots: {', '.join(target_strategies)}")
    print(f"Generating reports for {model_names}")

    # make sure that evaluations[i].superforecast is the same for all i, then use that value
    #   to determine if we should plot the calibration graph
    superforecast = evaluations[0].get("superforecast")
    superforecast_values = [e.get("superforecast") for e in evaluations]
    if not all(v == superforecast for v in superforecast_values):
        print("ERROR: superforecast values are not the same for all evaluations. Fix the code")
        return

    # XXX: correct the confidence scores for formatting issues. This is a
    #   terrible hack and I'm sorry. :-(
    # For scores between 0 and 1, leave them alone. For scores betweeen 25
    #   and 100, which are obviously misparsed percentages, divide them by 100.
    # The model is never so unconfident that a score between 1 and 25 would
    #   make sense, so give up if we see that.
    #del evaluations[333] # XXX

    for i, eval in enumerate(evaluations):
        for strategy in eval['confidence_results'].keys():
            for time_point in time_points:
                confidence = eval['confidence_results'][strategy][time_point + '_confidence']
                if confidence is None:
                    print(f"Warning: confidence is None for example {i}, strategy {strategy}, time_point {time_point}")
                    continue
                if confidence >= 25 and confidence <= 100:
                    eval['confidence_results'][strategy][time_point + '_confidence'] = confidence / 100
                elif confidence >= 0 and confidence <= 1:
                    pass
                else:
                    raise ValueError(f"Strange confidence score: {confidence} for example {i}")

    for strategy in target_strategies:
        print(f"Processing strategy: {strategy}")
        for time_point in time_points:
            print(f"  Processing time point: {time_point}")
            scores, outcomes = extract_confidence_data(evaluations, strategy, time_point)
            print(f"    Found {len(scores)} valid data points.")
            plot_calibration_graph(
                scores,
                outcomes,
                strategy,
                superforecast,
                time_point,
                model_names,
                run_output_dir,
                num_bins=args.bins
            )

    # Now, tabulate stats by various axes: each example has a language, a difficulty, and an original model. See how we did on each, and overall.
    # We don't need to draw a whole graph for each: just make tables.
    display_stats(evaluations, strategies)

    print("Report generation complete.")

def display_stats(evaluations, strategies):
    # Create a dictionary to store stats by language, difficulty, and original model
    stats = {
        'language': {},
        'difficulty': {},
        'original_model': {}
    }

    # Process each evaluation
    for eval in evaluations:
        language = eval['example']['language'].lower()
        difficulty = eval['example']['difficulty']
        original_model = eval['example']['original_model']

        if 'overall' not in stats:
            stats['overall'] = {
                'total': 0,
                'correct': 0,
                'incorrect': 0
            }

        # Initialize stats for this language, difficulty, and original model
        if language not in stats['language']:
            stats['language'][language] = {
                'total': 0,
                'correct': 0,
                'incorrect': 0
            }
        if difficulty not in stats['difficulty']:
            stats['difficulty'][difficulty] = {
                'total': 0,
                'correct': 0,
                'incorrect': 0
            }
        if original_model not in stats['original_model']:
            stats['original_model'][original_model] = {
                'total': 0,
                'correct': 0,
                'incorrect': 0
            }

        # Update stats for this example
        stats['overall']['total'] += 1
        stats['language'][language]['total'] += 1
        stats['difficulty'][difficulty]['total'] += 1
        stats['original_model'][original_model]['total'] += 1

        # Update correct/incorrect counts
        if eval['match']:
            stats['overall']['correct'] += 1
            stats['language'][language]['correct'] += 1
            stats['difficulty'][difficulty]['correct'] += 1
            stats['original_model'][original_model]['correct'] += 1
        else:
            stats['overall']['incorrect'] += 1
            stats['language'][language]['incorrect'] += 1
            stats['difficulty'][difficulty]['incorrect'] += 1
            stats['original_model'][original_model]['incorrect'] += 1

    # Now average confidence over all evaluations
    # evaluations[n]['confidence_results'][strategy]['before_confidence'] and ['after_confidence']
    # and display the average confidence overall
    confidence = [e['confidence_results'][strategy][time_point + '_confidence']
                  for e in evaluations
                  for strategy in strategies for time_point in time_points]
    print(f"Average confidence: {np.mean(confidence)}")

    # Now do the same but split by language:
    confidence_by_language = {
        language: [e['confidence_results'][strategy][time_point + '_confidence']
                  for e in evaluations
                  for strategy in strategies for time_point in time_points
                  if e['example']['language'].lower() == language.lower()]
        for language in stats['language'].keys()
    }
    for language in confidence_by_language.keys():
        print(f"Average confidence for {language}: {np.mean(confidence_by_language[language])}")

    # Display stats in a table format
    print("\nOverall Stats:")
    print(f"{'Total':<8} {'Correct':<8} {'Incorrect':<8} {'Accuracy':<8}")
    print("-" * 50)
    total = stats['overall']['total']
    correct = stats['overall']['correct']
    incorrect = stats['overall']['incorrect']
    accuracy = correct / total if total > 0 else 0
    print(f"{total:<8} {correct:<8} {incorrect:<8} {accuracy:.2%}")
    print("\nLanguage Stats:")
    print(f"{'Language':<15} {'Total':<8} {'Correct':<8} {'Incorrect':<8} {'Accuracy':<8}")
    print("-" * 50)
    for language in sorted(stats['language'].keys()):
        total = stats['language'][language]['total']
        correct = stats['language'][language]['correct']
        incorrect = stats['language'][language]['incorrect']
        accuracy = correct / total if total > 0 else 0
        print(f"{language:<15} {total:<8} {correct:<8} {incorrect:<8} {accuracy:.2%}")

    print("\nDifficulty Stats:")
    print(f"{'Difficulty':<15} {'Total':<8} {'Correct':<8} {'Incorrect':<8} {'Accuracy':<8}")
    print("-" * 50)
    for difficulty in sorted(stats['difficulty'].keys()):
        total = stats['difficulty'][difficulty]['total']
        correct = stats['difficulty'][difficulty]['correct']
        incorrect = stats['difficulty'][difficulty]['incorrect']
        accuracy = correct / total if total > 0 else 0
        print(f"{difficulty:<15} {total:<8} {correct:<8} {incorrect:<8} {accuracy:.2%}")

    print("\nOriginal Model Stats:")
    print(f"{'Original Model':<15} {'Total':<8} {'Correct':<8} {'Incorrect':<8} {'Accuracy':<8}")
    print("-" * 50)
    for model in sorted(stats['original_model'].keys()):
        total = stats['original_model'][model]['total']
        correct = stats['original_model'][model]['correct']
        incorrect = stats['original_model'][model]['incorrect']
        accuracy = correct / total if total > 0 else 0
        print(f"{model:<15} {total:<8} {correct:<8} {incorrect:<8} {accuracy:.2%}")

if __name__ == "__main__":
    main()
