import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from mpl_toolkits.mplot3d import Axes3D


plt.rcParams.update({'font.size': 25})


def normalize_metric_by_gamma(metrics, prop, metric_name="f1"):
    # Create a copy of the DataFrame to avoid modifying the original
    normalized_metrics = metrics.copy()

    # Get the 'scale' gamma values and index by the prop (e.g., c_value)
    scaled_mean = metrics[metrics['gamma'] == 'scale'].set_index(prop)[f"{metric_name}_mean"]
    scaled_std = metrics[metrics['gamma'] == 'scale'].set_index(prop)[f"{metric_name}_std"]

    # Apply normalization for both mean and standard deviation
    normalized_metrics[f'{metric_name}_mean_normalized'] = normalized_metrics.apply(
        lambda row: row[f'{metric_name}_mean'] / scaled_mean.loc[row[prop]], axis=1
    )
    normalized_metrics[f'{metric_name}_std_normalized'] = normalized_metrics.apply(
        lambda row: row[f'{metric_name}_std'] / scaled_mean.loc[row[prop]], axis=1
    )

    return normalized_metrics


def plot_normalized_gamma(metrics, metric_name='f1', prop='c_value'):
    # Normalize the metric values by scale
    normalized_metrics = normalize_metric_by_gamma(metrics, prop, metric_name)
    print_all_merics(normalized_metrics)

    # Create a scatter plot with error bars
    plt.figure(figsize=(12, 6), dpi=250)

    # Plot lines, points, and error bars for each gamma value
    for g in normalized_metrics['gamma'].unique():
        subset = normalized_metrics[normalized_metrics['gamma'] == g]

        # Sort by prop for smooth line joining
        subset = subset.sort_values(by=prop)

        # Plot with error bars
        # plt.errorbar(
        #     subset[prop],
        #     subset[f'{metric_name}_mean_normalized'],
        #     yerr=subset[f'{metric_name}_std_normalized'],
        #     fmt='o',  # Points as circles
        #     linestyle='-',  # Connect points with lines
        #     capsize=0,  # Add caps to error bars
        #     label=f'gamma={g}',
        #     alpha=0.7
        # )

        plt.scatter(subset[prop], subset[f'{metric_name}_mean_normalized'], label=f'Gamma = {g}')
        plt.plot(subset[prop], subset[f'{metric_name}_mean_normalized'])

    plt.xscale('log')  # Set the x-axis to a logarithmic scale
    plt.xlabel(prop)
    plt.ylabel(f'Normalized {metric_name} Mean (by scale)')
    # plt.title(f'Normalized {metric_name} Mean Comparison (Log Scale)')
    plt.legend(title='Gamma')
    plt.grid(True, which="both", ls="--")  # Gridlines for log scale
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show plot
    plt.show()


def plot_bar_metric_vs_prop(metrics, prop='c_value', metric_name='accuracy', param=None, filters=None, ylim=0.95):
    plt.figure(figsize=(12, 6), dpi=200)

    if filters is not None:
        for filter in filters:
            metrics = metrics[metrics[filter[0]] ==
                              filter[1]].reset_index(drop=True)

    # Use sns.barplot instead of scatterplot
    if param is None:
        metrics = metrics[[prop, f"{metric_name}_mean", f"{metric_name}_std"]]
        ax = sns.barplot(data=metrics, x=prop, y=f'{
                         metric_name}_mean', palette='muted', ci=None)
    else:
        metrics = metrics.sort_values(by=[prop, param], ascending=[True, True]).reset_index(drop=True)
        metrics = metrics[[prop, param, f"{
            metric_name}_mean", f"{metric_name}_std"]]
        ax = sns.barplot(data=metrics, x=prop, y=f'{
                         metric_name}_mean', hue=param, palette='muted', ci=None)

    # print_all_merics(metrics)

    # Get positions of the bars
    bar_positions = [p.get_x() + p.get_width() / 2 for p in sorted(ax.patches[:len(metrics)], key=lambda p: p.get_x())]
    print(bar_positions)
    # Add error bars at the bar positions
    for i, bar_pos in enumerate(bar_positions):
        print("i:", i)
        print(metrics.iloc[i])
        print("\nbar_pos", bar_pos,
              "\nmean", metrics[f'{metric_name}_mean'][i],
              "\nstd:", metrics[f'{metric_name}_std'][i])

        plt.errorbar(x=bar_pos,
                     y=metrics[f'{metric_name}_mean'][i],
                     yerr=metrics[f'{metric_name}_std'][i],
                     fmt='none', c='black', capsize=3)

    filter_text = ""
    if filters is not None:
        for filter in filters:
            filter_text = f"{filter_text} with {filter[0]} = {filter[1]}"

    if param is None:
        plt.title(f'{metric_name.capitalize()} by {prop}{filter_text}')
    else:
        # plt.title(f'{metric_name.capitalize()} by {
        #          prop} and {param}{filter_text}')
        # plt.legend(title=f"{param}", bbox_to_anchor=(
        #     1.05, 1), loc='upper left')

        plt.legend(title="Grado", bbox_to_anchor=(
            1.05, 1), loc='upper left')

    plt.xlabel(prop)
    plt.ylim(ylim, 1)
    plt.ylabel(f'{metric_name.capitalize()}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_scatter_metric_vs_prop(metrics, prop='c_value', metric_name='accuracy', param=None, filters=None, ylim=0.95):
    plt.figure(figsize=(12, 6), dpi=250)

    if filters is not None:
        for filter in filters:
            metrics = metrics[metrics[filter[0]] == filter[1]].reset_index(drop=True)

    # Scatter plot instead of barplot
    if param is None:
        metrics = metrics[[prop, f"{metric_name}_mean", f"{metric_name}_std"]]
        plt.scatter(metrics[prop], metrics[f"{metric_name}_mean"], s=50, label=prop, c='b', linestyle='-')
        plt.plot(metrics[prop], metrics[f"{metric_name}_mean"], 'bo-', label=prop)  # 'b' for blue color, 'o-' for points connected by lines

        # Add error bars
        plt.errorbar(metrics[prop], metrics[f"{metric_name}_mean"], 
                     yerr=metrics[f"{metric_name}_std"], fmt='none', c='black', capsize=3, linestyle='-')

    else:
        metrics = metrics.sort_values(by=[prop, param], ascending=[True, True]).reset_index(drop=True)
        metrics = metrics[[prop, param, f"{metric_name}_mean", f"{metric_name}_std"]]

        # Loop over unique param values for coloring different groups
        unique_params = metrics[param].unique()
        for p in unique_params:
            sub_data = metrics[metrics[param] == p]
            plt.scatter(sub_data[prop], sub_data[f"{metric_name}_mean"], s=100, label=f"{param}: {p}")

            # Add error bars
            plt.errorbar(sub_data[prop], sub_data[f"{metric_name}_mean"], 
                         yerr=sub_data[f"{metric_name}_std"], fmt='none', c='black', capsize=3)

    # Print metrics for debugging
    print_all_merics(metrics)

    # Add titles and labels
    filter_text = ""
    if filters is not None:
        for filter in filters:
            filter_text = f"{filter_text} with {filter[0]} = {filter[1]}"

    if param is None:
        # plt.title(f'{metric_name.capitalize()} by {prop}{filter_text}')
        print("Title Commented")
    else:
        plt.title(f'{metric_name.capitalize()} by {prop} and {param}{filter_text}')
        plt.legend(title=f"{param}", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xscale('log')
    plt.xlabel(prop)
    plt.ylim(ylim, 1)
    plt.ylabel(f'{metric_name.capitalize()}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def print_all_merics(metrics):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(metrics)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')


def main():
    parser = argparse.ArgumentParser(description='Process metrics CSV file.')
    parser.add_argument('base_dir', type=str,
                        help='Path to the base dir')
    parser.add_argument('kernel', type=str,
                        help='Kernel')
    parser.add_argument('final', type=str,
                        help='Final if its the final csv', default="not_final")
    args = parser.parse_args()

    base_dir = args.base_dir
    kernel = args.kernel
    final = args.final

    kernel_dir = os.path.join(base_dir, kernel)

    if final == "final":
        file_path = os.path.join(kernel_dir, "metrics_final.csv")
    else:
        file_path = os.path.join(kernel_dir, "metrics.csv")

    df = pd.read_csv(file_path, sep=';')

    if kernel == 'linear':
        groupby_cols = ['kernel', 'c_value']
    elif kernel == 'poly':
        groupby_cols = ['kernel', 'c_value', 'gamma', 'degree', 'coef0']
    elif kernel == 'rbf':
        groupby_cols = ['kernel', 'c_value', 'gamma']
    elif kernel == 'sigmoid':
        groupby_cols = ['kernel', 'c_value', 'gamma', 'coef0']

    metrics = df.groupby(groupby_cols)[
        ['precision', 'recall', 'f1', 'accuracy']].agg(['mean', 'std'])

    if final == "final":
        metrics.columns = ['_'.join(col).strip()
                           for col in metrics.columns.values]
        metrics = metrics.reset_index()
        cols = groupby_cols
        cols.extend(['precision_mean', 'precision_std',
                     'recall_mean', 'recall_std',
                     'f1_mean', 'f1_std',
                     'accuracy_mean', 'accuracy_std'])
        metrics.columns = cols
    else:
        count = df.groupby(groupby_cols)[
            'iteration'].count().reset_index(name='count')

        metrics.columns = ['_'.join(col).strip()
                           for col in metrics.columns.values]
        metrics = metrics.reset_index()
        metrics = metrics.merge(count, on=groupby_cols)
        cols = groupby_cols
        cols.extend(['precision_mean', 'precision_std',
                     'recall_mean', 'recall_std',
                     'f1_mean', 'f1_std',
                     'accuracy_mean', 'accuracy_std',
                     'data_count'])
        metrics.columns = cols

    output_file_path = os.path.join(kernel_dir, "out.csv")
    metrics.to_csv(output_file_path, index=False, sep=';')

    sns.set(style="whitegrid")

    if kernel == 'poly':
        # Necesita ponerse 2 filtros por la cantidad de parametros que recibe
        plot_bar_metric_vs_prop(metrics, metric_name='f1', prop='c_value', param="degree", filters=[
                                ('gamma', "10"), ('coef0', 0)])

        # plot_bar_metric_vs_prop(metrics, metric_name='f1', prop='degree', param="c_value", filters=[
        #                         ('gamma', "10"), ('coef0', 1)])

        metric_name = "f1"
        sorted_metrics = metrics.sort_values(by=f"{metric_name}_mean", ascending=False).reset_index(drop=True)
        sorted_metrics = sorted_metrics[["c_value", "degree", "gamma", "coef0", f"{metric_name}_mean", f"{metric_name}_std"]]
        print_all_merics(sorted_metrics)

    elif kernel == 'sigmoid':
        # Necesita minimo un filtro
        plot_bar_metric_vs_prop(metrics, metric_name='f1', prop='gamma',
                                param="c_value", filters=[('coef0', 0)], ylim=0)
    elif kernel == 'rbf':
        # No neceita filtros
        # plot_bar_metric_vs_prop(metrics, metric_name='f1',
        #                         prop='c_value', param="gamma", ylim=0.97)

        plot_normalized_gamma(metrics, metric_name='f1', prop='c_value')

    elif kernel == 'linear':
        # No neceita filtros ni param
        plot_scatter_metric_vs_prop(metrics, metric_name='f1', prop='c_value')

    exit()


main()
