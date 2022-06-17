from gerador_amostras import Amostra, GeradorDeAmostras
from gerenciador_arquivos import GerenciadorArquivos

gerenciador = GerenciadorArquivos("dados/caracteres-limpo.csv")
amostrador = GeradorDeAmostras()

base_caracteres = gerenciador.le_csv(df_caracteres=True, salvar=False)
X_caracteres, y_caracteres = gerenciador.extrai_X_y(base_caracteres, True)

amostra_holdout = amostrador.holdout(X_caracteres, y_caracteres)
amostra_k_fold = amostrador.k_fold(X_caracteres, y_caracteres)

def escrever_dataset(file, fold):
    # conjunto => [ treinamento, validacao ]
    for conjunto in fold:
        x, y = conjunto

        for index in range(len(x)):
            elementos = x[index] + y[index]
            tamanho = len(elementos)

            for indice in range(0, tamanho - 1):
                file.write(f"{elementos[indice]},")
            file.write(f"{elementos[tamanho - 1]}\n")


# Escreve os 21 valores (14 primeiros de treinamento e 7 de validação)
with open("./validando-holdout-final.csv", mode="w", encoding="utf-8") as file:
    for fold in amostra_holdout.folds:
        escrever_dataset(file, fold)

# Escreve 63 valores (Cada conjunto de 21 se refere a um fold)
with open("./validando-kfold-final.csv", mode="w", encoding="utf-8") as file:
    for fold in amostra_k_fold.folds:
        escrever_dataset(file, fold)

