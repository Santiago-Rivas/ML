import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

plt.rcParams.update({'font.size': 25})

parser = argparse.ArgumentParser(description='Process metrics CSV file.')
parser.add_argument('input_file', type=str, help='Path to the input metrics CSV file')
parser.add_argument('output_file', type=str, help='Path to the output metrics summary CSV file')
args = parser.parse_args()

file_path = args.input_file
df = pd.read_csv(file_path, sep=';')

metrics = df.groupby(['kernel', 'c_value', 'gamma'])[
    ['precision', 'recall', 'f1', 'accuracy']].agg(['mean', 'std'])

count = df.groupby(['kernel', 'c_value', 'gamma'])[
    'iteration'].count().reset_index(name='count')

metrics.columns = ['_'.join(col).strip() for col in metrics.columns.values]
metrics = metrics.reset_index()
metrics = metrics.merge(count, on=['kernel', 'c_value', 'gamma'])

metrics.columns = ['kernel', 'c_value', 'gamma',
                   'precision_mean', 'precision_std',
                   'recall_mean', 'recall_std',
                   'f1_mean', 'f1_std',
                   'accuracy_mean', 'accuracy_std',
                   'data_count']

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print(metrics)

pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

output_file_path = args.output_file
metrics.to_csv(output_file_path, index=False, sep=';')

sns.set(style="whitegrid")


def plot_metric(metrics, metric_name):
    plt.figure(figsize=(12, 6), dpi=250)
    metrics = metrics.sort_values(by=['c_value', 'kernel'], ascending=[True, True]).reset_index(drop=True)

    ax = sns.barplot(data=metrics, x='c_value', y=f'{metric_name}_mean', hue='kernel',
                palette='muted', ci=None)

    # for i in range(len(metrics)):
    #     plt.errorbar(x=i,  # bar position
    #                  y=metrics[f'{metric_name}_mean'][i],
    #                  yerr=metrics[f'{metric_name}_std'][i],
    #                  fmt='none', c='black', capsize=3)


    bar_positions = [p.get_x() + p.get_width() / 2 for p in sorted(ax.patches[:len(metrics)], key=lambda p: p.get_x())]

    for i, bar_pos in enumerate(bar_positions):
        plt.errorbar(x=bar_pos,
                     y=metrics[f'{metric_name}_mean'][i],
                     yerr=metrics[f'{metric_name}_std'][i],
                     fmt='none', c='black', capsize=3)
    plt.xscale('log')

    # plt.title(f'{metric_name.capitalize()} by Kernel and C Parameter')
    # plt.xlabel('C Parameter')
    # plt.ylabel(f'{metric_name.capitalize()}')
    plt.legend(title='Kernel', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


for metric in ['accuracy']:
    metrics = metrics[metrics['kernel'] != 'sigmoid']
    plot_metric(metrics, metric)
