import json
from time import time

from gerenciador_arquivos import GerenciadorArquivos
from visualizacao import plota_mapa_de_calor_X_y
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
gerenciador = GerenciadorArquivos("dados/caracteres-limpo.csv")
base_caracteres = gerenciador.le_csv(df_caracteres=True)
X_caracteres, y_caracteres = gerenciador.extrai_X_y(base_caracteres, True)
print(f"Variáveis X_caracteres \n {X_caracteres}", "\n")
print(f"Variável y_caracteres \n {y_caracteres}", "\n")

#Contabilizando tempo de treinamento
inicio = time()
 
gerenciador_logs = GerenciadorLogs()
 
modelo = MultilayerPerceptron(len(X_caracteres[0]), 15, 7, tx_aprendizagem=0.1, gerenciador_logs=gerenciador_logs)
modelo.treina(X_caracteres, y_caracteres, epocas=100)

fim = time()
tempo_treinamento = fim - inicio
print(f"Tempo de treinamento: {tempo_treinamento}")

log = gerenciador_logs.log_completo()
print(log)

log_html = gerenciador_logs.log_html()
#salvando html 
with open("log_html.html", "w") as f:
    f.write(log_html)


predito = modelo.prediz(X_caracteres)
print(f"Predito \n {predito}", "\n")

# Gravando o modelo como json
json_modelo = modelo.to_json()
with open("modelo.json", "w") as json_file:
    json_file.write(json.dumps(json_modelo))

# Testando se o modelo é igual ao que foi salvo
with open("modelo.json", "r") as json_file:
    json_lido = json.loads(json_file.read())
    modelo_salvo = MultilayerPerceptron.from_json(json_lido)

print("Modelo salvo: \n", modelo_salvo.to_json())
