import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

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


def plot_metric(metric_name):
    plt.figure(figsize=(12, 6), dpi=150)

    sns.scatterplot(data=metrics, x='c_value', y=f'{metric_name}_mean', hue='kernel',
                    s=100, palette='muted', style='kernel', markers=['o', 's', 'D'])

    for i in range(len(metrics)):
        plt.errorbar(x=metrics['c_value'][i],
                     y=metrics[f'{metric_name}_mean'][i],
                     yerr=metrics[f'{metric_name}_std'][i],
                     fmt='none', c='black', capsize=3)
    plt.xscale('log')

    plt.title(f'{metric_name.capitalize()} by Kernel and c_value')
    plt.xlabel('C Value')
    plt.ylabel(f'{metric_name.capitalize()}')
    plt.legend(title='Kernel', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


for metric in ['precision', 'recall', 'f1', 'accuracy']:
    plot_metric(metric)
