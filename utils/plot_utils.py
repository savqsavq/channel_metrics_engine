import matplotlib.pyplot as plt
import seaborn as sns

def scatter_plot(x, y, title, fname):
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=x, y=y, alpha=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()

def bar_plot(labels, values, title, fname):
    plt.barh(labels[::-1], values[::-1], color="#6688cc")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()