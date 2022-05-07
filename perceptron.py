import math

class Perceptron(object):
    def __init__(self, pesos: dict, bias: float):
        self.pesos = pesos
        self.bias = bias
    

    def funcao_ativacao(self, y_in: float) -> float:
        """ Retorna o valor da função de ativação para o valor y_in """
        return 1 / (1 + math.exp(-y_in))

    def train(self, X: list, y: list, epochs: int = 10, learning_rate: float = 0.1, df_caracteres: bool = False, indice_char: int = None):
        """ Treina o perceptron com os dados de entrada X e saída y de acordo com os parâmetros especificados"""