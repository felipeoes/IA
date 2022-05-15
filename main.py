from gerenciador_arquivos import GerenciadorArquivos
from visualizacao import plota_mapa_de_calor_X_y
from multilayer_perceptron import MultilayerPerceptron

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

modelo = MultilayerPerceptron(len(X_caracteres[0]), 15, 7, tx_aprendizagem=0.1)
modelo.treina(X_caracteres, y_caracteres, epocas=10000)
predito = modelo.prediz(X_caracteres)
print(f"Predito \n {predito}", "\n")
