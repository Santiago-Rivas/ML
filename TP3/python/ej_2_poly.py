import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from mpl_toolkits.mplot3d import Axes3D


def plot_bar_metric_vs_prop(metrics, prop='c_value', metric_name='accuracy', param=None, filters=None, ylim=0.95):
    plt.figure(figsize=(12, 6), dpi=150)

    if filters is not None:
        for filter in filters:
            metrics = metrics[metrics[filter[0]] ==
                              filter[1]].reset_index(drop=True)

    # print_all_merics(metrics)
    # Use sns.barplot instead of scatterplot
    if param is None:
        ax = sns.barplot(data=metrics, x=prop, y=f'{
                         metric_name}_mean', palette='muted', ci=None)
    else:
        ax = sns.barplot(data=metrics, x=prop, y=f'{
                         metric_name}_mean', hue=param, palette='muted', ci=None)

    # Get positions of the bars
    bar_positions = [p.get_x() + p.get_width() /
                     2 for p in ax.patches[:len(metrics)]]

    # Add error bars at the bar positions
    for i, bar_pos in enumerate(bar_positions):
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
        plt.title(f'{metric_name.capitalize()} by {
                  prop} and {param}{filter_text}')
        plt.legend(title=f"{param}", bbox_to_anchor=(
            1.05, 1), loc='upper left')

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
        plot_bar_metric_vs_prop(metrics, metric_name='f1', prop='c_value', param="gamma", filters=[
                                ('degree', 1), ('coef0', 0)], ylim=0.9)
    elif kernel == 'sigmoid':
        # Necesita minimo un filtro
        plot_bar_metric_vs_prop(metrics, metric_name='f1', prop='gamma', param="c_value", filters=[('coef0', 0)], ylim=0)
    elif kernel == 'rbf':
        # No neceita filtros
        plot_bar_metric_vs_prop(metrics, metric_name='f1', prop='gamma', param="c_value")
    elif kernel == 'linear':
        # No neceita filtros ni param
        plot_bar_metric_vs_prop(metrics, metric_name='f1', prop='c_value')

    exit()


main()
