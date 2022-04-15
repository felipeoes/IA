from arquivos import le_csv, extrai_X_y
from visualizacao import plota_mapa_de_calor_X_y
from multilayer_perceptron import MultilayerPerceptron

base_AND = le_csv('dados/problemAND.csv',
                  nomes_cols=['expressao0', 'expressao1', 'resultado'])
X_AND, y_AND = extrai_X_y(base_AND)

print(f"Variáveis X_AND \n {X_AND}", "\n")

print(f"Variável y_AND \n {y_AND}", "\n")

base_caracteres = le_csv('dados/caracteres-limpo.csv', df_caracteres=True)
X_caracteres, y_caracteres = extrai_X_y(base_caracteres, True)
print(f"Variáveis X_caracteres \n {X_caracteres}", "\n")
print(f"Variável y_caracteres \n {y_caracteres}", "\n")

plota_mapa_de_calor_X_y(base_caracteres, len(
    X_caracteres.columns), 'Matriz de correlação dos caracteres')

modelo_mp = MultilayerPerceptron(X_caracteres, y_caracteres)
