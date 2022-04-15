import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plota_mapa_de_calor(base: pd.DataFrame, title="Matriz de correlação", mapa_cores="coolwarm", figsize=(10, 10)):
    sns.set(style="white")
    """ Plota um mapa de calor da correlação entre as variáveis do dataframe 'base' """

    corr = base.corr()

    # Configurando a figura
    f, ax = plt.subplots(figsize=figsize)

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title(title)
    plt.show()


def plota_mapa_de_calor_X_y(base: pd.DataFrame, tam_X: int, title="Matriz de correlação", figsize=(10, 10)):
    sns.set(style="white")
    """ Plota um mapa de calor da correlação entre as variáveis X e a variável target do dataframe 'base' """

    corr = base.corr()
    cols_y = [col for col in base.columns if col.startswith("rotulo")]

    X_y = corr.loc[[*cols_y]]

    # Configurando a figura
    f, ax = plt.subplots(figsize=figsize)

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(X_y, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title(title)
    plt.xlabel("Rótulos")
    plt.ylabel("Atributos")
    plt.show()
