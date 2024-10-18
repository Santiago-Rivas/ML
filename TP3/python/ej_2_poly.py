import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(description='Process metrics CSV file.')
parser.add_argument('input_file', type=str, help='Path to the input metrics CSV file')
parser.add_argument('output_file', type=str, help='Path to the output metrics summary CSV file')
parser.add_argument('final', type=str, help='Path to the output metrics summary CSV file', default="not_final")
args = parser.parse_args()

file_path = args.input_file
df = pd.read_csv(file_path, sep=';')
final = args.final
metrics = df.groupby(['kernel', 'c_value', 'gamma', 'degree'])[
    ['precision', 'recall', 'f1', 'accuracy']].agg(['mean', 'std'])


if final == "final":
    # count = df.groupby(['kernel', 'c_value', 'gamma', 'degree'])[
        # 'iteration'].count().reset_index(name='count')

    metrics.columns = ['_'.join(col).strip() for col in metrics.columns.values]
    metrics = metrics.reset_index()
    # metrics = metrics.merge(count, on=['kernel', 'c_value', 'gamma', 'degree'])

    metrics.columns = ['kernel', 'c_value', 'gamma', 'degree',
                       'precision_mean', 'precision_std',
                       'recall_mean', 'recall_std',
                       'f1_mean', 'f1_std',
                       'accuracy_mean', 'accuracy_std']
                       # 'data_count']
else:
    count = df.groupby(['kernel', 'c_value', 'gamma', 'degree'])[
        'iteration'].count().reset_index(name='count')

    metrics.columns = ['_'.join(col).strip() for col in metrics.columns.values]
    metrics = metrics.reset_index()
    metrics = metrics.merge(count, on=['kernel', 'c_value', 'gamma', 'degree'])

    metrics.columns = ['kernel', 'c_value', 'gamma', 'degree',
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


def plot_metric_degree(metric_name, param='c_value'):
    plt.figure(figsize=(12, 6), dpi=150)

    sns.scatterplot(data=metrics, x='degree', y=f'{metric_name}_mean', hue=param,
                    s=100, palette='muted', style=param, markers=['o', 's', 'D'])

    for i in range(len(metrics)):
        plt.errorbar(x=metrics['degree'][i],
                     y=metrics[f'{metric_name}_mean'][i],
                     yerr=metrics[f'{metric_name}_std'][i],
                     fmt='none', c='black', capsize=3)

    plt.title(f'{metric_name.capitalize()} by degree and {param}')
    plt.xlabel("Degree")
    plt.ylabel(f'{metric_name.capitalize()}')
    plt.legend(title=f"{param}", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


plot_metric_degree('f1')
exit()

for metric in ['precision', 'recall', 'f1', 'accuracy']:
    plot_metric_degree(metric)
