import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

LETRAS = ["A", "B", "C", "D", "E", "J", "K", "L", "V", "Y", "Z"]


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


def calcula_matriz_confusao(y_real: list, y_pred: list):
    """ Calcula a matriz de confusão  dados os vetores de real e predição """
    # Quantidade de classes do problema
    qtde_classes = len(set(y_real))

    # Inicializa a matriz de confusão com zeros
    matriz = [[0 for _ in range(qtde_classes)] for _ in range(qtde_classes)]

    # Para cada letra da lista de real, busca a letra correspondente no vetor de predição
    for letra_real, letra_pred in zip(y_real, y_pred):
        indice_real = LETRAS.index(letra_real)
        indice_pred = LETRAS.index(letra_pred)

        # Incrementa o valor da posição da matriz de confusão
        matriz[indice_real][indice_pred] += 1

    return matriz


def plota_matriz_de_confusao(y_real: list = None, y_pred: list = None, title: str = "Matriz de confusão", figsize=(10, 10), matriz: list[list] = None):
    sns.set(style="white")
    """ Plota a matriz de confusão """

    if not matriz:
        # Gera a matriz de confusão a partir do vetor predito e real
        matriz = calcula_matriz_confusao(y_real, y_pred)

    classes = sorted(set(y_real))
    df_matriz = pd.DataFrame(matriz, index=classes, columns=classes)

    graf = sns.heatmap(df_matriz, annot=True, fmt='d', cbar=False)
    plt.title(title)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.show()

    dir_img = "./saidas/imagens/matriz_confusao.png"
    graf.figure.savefig(dir_img)  # Salvando a imagem em formato png


def decodifica_vetor_predicao(predicoes: list):
    """ Decodifica o vetor de predição para uma lista de letras """
    letras = []
    for predicao in predicoes:
        # Busca o índice do maximo valor do vetor
        indice_max = predicao.index(max(predicao))

        # Adiciona a letra correspondente ao índice
        letras.append(LETRAS[indice_max])

    return letras
