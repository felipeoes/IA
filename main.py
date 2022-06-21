import json

from time import time
from datetime import datetime

from gerenciador_arquivos import GerenciadorArquivos
# from visualizacao import plota_mapa_de_calor_X_y
from multilayer_perceptron import MultilayerPerceptron
from gerenciador_logs import GerenciadorLogs

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
gerenciador = GerenciadorArquivos("dados/caracteres-limpo.csv", tamanho_entrada=7)
# gerenciador = GerenciadorArquivos("dados/caracteres-ruido-11-letras.csv", tamanho_entrada=11)
base_caracteres = gerenciador.le_csv(df_caracteres=True, salvar=False)
X_caracteres, y_caracteres = gerenciador.extrai_X_y(base_caracteres, True)

# for index in range(len(X_caracteres)):
#     print(f"A linha {index} tem x: ${len(X_caracteres[index])} e y: ${len(y_caracteres[index])}")
#     print(X_caracteres[index])
#     print(y_caracteres[index])
#     print("\n")
#     if index % 10 == 0:
#         print("\n" * 5)

# Contabilizando tempo de treinamento
inicio = time()

gerenciador_logs = GerenciadorLogs("./modelo-7-ruido.xlsx")

modelo = MultilayerPerceptron(len(
    X_caracteres[0]), 10, 7, tx_aprendizagem=0.1, gerenciador_logs=gerenciador_logs)

modelo.treina(X_caracteres, y_caracteres, epocas=50_000)

fim = time()
tempo_treinamento = fim - inicio
print(f"Tempo de treinamento: {tempo_treinamento}")

# log = gerenciador_logs.log_completo()
# print(log['Erro Epoca'])
# gerenciador_logs.salva_log_excel()

# log_html = gerenciador_logs.log_html()
# # salvando html
# with open("log_html_11_letras.html", "w") as f:
#     f.write(log_html)

# predito = modelo.prediz(X_caracteres)
# print(f"Predito \n {predito}", "\n")

#Gravando o modelo como json
json_modelo = modelo.to_json()

# Para não sobreescrever sem querer
now = datetime.today()
path_modelo = f"./modelos/modelo-{now.strftime('%d-%m-%Y-%H-%M-%S')}.json"

with open(path_modelo, mode="w") as json_file:
    json_file.write(json.dumps(json_modelo))

# # Testando se o modelo é igual ao que foi salvo

gerenciador_ruidos = GerenciadorArquivos("dados/caracteres-ruido.csv", tamanho_entrada=7)
base_caracteres_ruidos = gerenciador_ruidos.le_csv(df_caracteres=True, salvar=False)
X_caracteres_ruido, y_caracteres_ruido = gerenciador_ruidos.extrai_X_y(base_caracteres_ruidos, True)


matriz = modelo.gera_matriz_de_confusao(X_caracteres_ruido, y_caracteres_ruido)

print(matriz)

# with open("./modelos/modelo-11-letras.json", "r") as json_file:
    # json_lido = json.loads(json_file.read())
    # modelo_salvo = MultilayerPerceptron.from_json(json_lido)
    # print(modelo_salvo)
    # matriz = modelo_salvo.gera_matriz_de_confusao(X_caracteres, y_caracteres)
    # matriz = modelo_salvo.gera_matriz_de_confusao(X_caracteres_ruido, y_caracteres_ruido)
    # print(matriz)

# print("Modelo salvo: \n", modelo_salvo.to_json())
