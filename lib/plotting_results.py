import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df = pd.read_csv('../pics/results.csv')
    df = pd.melt(df, id_vars=['model', 'criterion', 'epoch'], value_vars=['train_loss', 'valid_loss', 'metric'])
    g = sns.FacetGrid(data=df, row='criterion', col='variable', margin_titles=True, sharey=False)
    g.map_dataframe(sns.pointplot, x='epoch', y='value', hue='model')
    g.add_legend()
    g.savefig('../results/results.png')
    plt.close()
    df2 = df.groupby(['model', 'criterion', 'variable']).agg({'value': max}).query('variable == "metric"').reset_index()
    ax = sns.barplot(data=df2, x='model', y='value', hue='criterion', palette='husl')
    sns.move_legend(ax, 'center right', bbox_to_anchor=(1.5, 0.5))
    ax.set(ylim=(df2['value'].min() * 0.97, df2['value'].max() * 1.01))
    plt.title('Comparison of models/loss', fontsize=16)
    plt.xlabel('Models', fontsize=16);
    plt.ylabel('IoU', fontsize=16);
    plt.savefig('../results/metrics.png', bbox_inches='tight', pad_inches=0.45)