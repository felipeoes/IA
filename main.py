import json

from time import time
from datetime import datetime
from gerenciador_arquivos import GerenciadorArquivos
from multilayer_perceptron import MultilayerPerceptron
from gerenciador_logs import GerenciadorLogs
from gerenciador_notebook import GerenciadorNotebook
from visualizacao import calcula_matriz_confusao, plota_matriz_de_confusao, decodifica_vetor_predicao

# gerenciador = GerenciadorArquivos("dados/problemAND.csv")
# base_AND = gerenciador.le_csv(
#     nomes_cols=['expressao0', 'expressao1', 'resultado'])
# X_AND, y_AND = gerenciador.extrai_X_y(base_AND)

# print(f"Variáveis X_AND \n {X_AND}", "\n")

# print(f"Variável y_AND \n {y_AND}", "\n")

# plota_mapa_de_calor_X_y(base_caracteres, len(
#     X_caracteres.columns), 'Matriz de correlação dos caracteres')

# PROBLEMA XOR
# gerenciador = GerenciadorArquivos("dados/problemXOR.csv")
# base_XOR = gerenciador.le_csv(
#     nomes_cols=['expressao0', 'expressao1', 'resultado'])
# X_XOR, y_XOR = gerenciador.extrai_X_y(base_XOR)
# print(f"Variáveis X_XOR \n {X_XOR}", "\n")
# print(f"Variável y_XOR \n {y_XOR}", "\n")

# modelo = MultilayerPerceptron(2, 4, 1, tx_aprendizagem=0.1)
# modelo.treina(X_XOR, y_XOR, epocas=10000)
# predito = modelo.prediz(X_XOR)
# print(f"Predito \n {predito}", "\n")


# PROBLEMA CARACTERES
gerenciador = GerenciadorArquivos(
    "dados/caracteres-limpo-11-letras.csv", tamanho_entrada=11)
base_caracteres = gerenciador.le_csv(df_caracteres=True, salvar=False)
X_caracteres, y_caracteres = gerenciador.extrai_X_y(base_caracteres, True)


# Contabilizando tempo de treinamento
inicio = time()

gerenciador_logs = GerenciadorLogs("./modelo-7-ruido.xlsx")
gerenciador_notebook = GerenciadorNotebook(gerenciador_logs)
modelo = MultilayerPerceptron(len(
    X_caracteres[0]), 20, 11, tx_aprendizagem=0.1, gerenciador_logs=gerenciador_logs)

# Separa a base de treinamento e validação
X_treino, y_treino, X_validacao, y_validacao = gerenciador.separa_base_treinamento_validacao(
    X_caracteres, y_caracteres, percent_treino=1)

modelo.treina(X_treino, y_treino, epocas=50_000)

fim = time()
tempo_treinamento = fim - inicio
print(f"Tempo de treinamento: {tempo_treinamento}")

# Gravando o modelo como json
json_modelo = modelo.to_json()

# Para não sobreescrever sem querer
now = datetime.today()
path_modelo = f"./modelos/modelo-{now.strftime('%d-%m-%Y-%H-%M-%S')}.json"

with open(path_modelo, mode="w") as json_file:
    json_file.write(json.dumps(json_modelo))

gerenciador_ruidos = GerenciadorArquivos(
    "dados/caracteres-ruido-11-letras-total.csv", tamanho_entrada=11)
base_caracteres_ruidos = gerenciador_ruidos.le_csv(
    df_caracteres=True, salvar=False)
X_caracteres_ruido, y_caracteres_ruido = gerenciador_ruidos.extrai_X_y(
    base_caracteres_ruidos, True)

# matriz = modelo.gera_matriz_de_confusao(X_caracteres_ruido, y_caracteres_ruido)
# print(matriz)

predito = modelo.prediz(X_caracteres_ruido)

# Decodifica os vetores de predição para caracteres
letras_preditas = decodifica_vetor_predicao(predito)
letras_reais = decodifica_vetor_predicao(y_caracteres_ruido)
plota_matriz_de_confusao(letras_reais, letras_preditas, "Matriz de confusão")

gerenciador_notebook.conjuntos = {
    'treinamento': X_treino,
    'validacao': X_validacao,
    'teste': X_caracteres_ruido
}

gerenciador_notebook.matriz_confusao = calcula_matriz_confusao(letras_reais, letras_preditas)

gerenciador_notebook.predicoes = [ f"Predito: {letras_preditas[index]} - Real: {letras_reais[index]} " for index in range(len(X_caracteres_ruido))]

gerenciador_notebook.grava_notebook(f"{now.strftime('%d-%m-%Y-%H-%M-%S')}")