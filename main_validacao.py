import json

from time import time
from datetime import datetime
from gerenciador_arquivos import GerenciadorArquivos
from multilayer_perceptron import MultilayerPerceptron
from gerenciador_logs import GerenciadorLogs
from gerenciador_notebook import GerenciadorNotebook
from visualizacao import calcula_matriz_confusao, plota_matriz_de_confusao, decodifica_vetor_predicao

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
    X_caracteres, y_caracteres, percent_treino=0.8)

modelo.treina(X_treino, y_treino, epocas=5000, conj_validacao=(X_validacao, y_validacao))

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

print("\n\n\n")
print(f"Notebook Dataset: http://localhost:8000/notebook/visualizar-dataset.html?report=notebook-{now.strftime('%d-%m-%Y-%H-%M-%S')}")
print(f"Notebook: http://localhost:8000/notebook/?report=notebook-{now.strftime('%d-%m-%Y-%H-%M-%S')}")