from gerenciador_arquivos import GerenciadorArquivos
from visualizacao import plota_mapa_de_calor_X_y
from multilayer_perceptron import MultilayerPerceptron

gerenciador = GerenciadorArquivos("dados/problemAND.csv")
base_AND = gerenciador.le_csv(
    nomes_cols=['expressao0', 'expressao1', 'resultado'])
X_AND, y_AND = gerenciador.extrai_X_y(base_AND)

print(f"Variáveis X_AND \n {X_AND}", "\n")

print(f"Variável y_AND \n {y_AND}", "\n")

gerenciador = GerenciadorArquivos("dados/caracteres-limpo.csv")
base_caracteres = gerenciador.le_csv(df_caracteres=True)
X_caracteres, y_caracteres = gerenciador.extrai_X_y(base_caracteres, True)
print(f"Variáveis X_caracteres \n {X_caracteres}", "\n")
print(f"Variável y_caracteres \n {y_caracteres}", "\n")

plota_mapa_de_calor_X_y(base_caracteres, len(
    X_caracteres.columns), 'Matriz de correlação dos caracteres')

modelo_mp = MultilayerPerceptron(X_caracteres, y_caracteres)
