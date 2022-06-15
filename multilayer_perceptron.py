import random
from perceptron import Perceptron
from gerenciador_logs import GerenciadorLogs


class CamadaMLP(object):
    """ Classe base para representar uma camada de uma rede neural multilayer perceptron"""

    def __init__(self, n_neuronios: int):
        self.n_neuronios = n_neuronios
        self.neuronios = []

    def inicializa_pesos(self, n_observacoes: int, tx_aprendizagem: float):
        """ Inicializa os pesos dos neuronios da camada de acordo com o número de observações na base de treino """
        for _ in range(self.n_neuronios):
            pesos = {col_num: random.uniform(-0.1, 0.9)
                     for col_num in range(n_observacoes)}
            bias = random.random()
            self.neuronios.append(Perceptron(pesos, bias, tx_aprendizagem))

    def calcula_saida(self, X: list):
        """ Calcula a saída da camada de acordo com os valores de entrada X"""
        saidas = []  # lista da saída de cada neuronio da camada
        valores_ins = []  # lista da soma ponderada de cada neuronio da camada

        for neuronio in self.neuronios:
            valor_in, saida = neuronio.calcula_saida(X)
            saidas.append(saida)
            valores_ins.append(valor_in)
        return valores_ins, saidas


class CamadaEntrada(CamadaMLP):
    def __init__(self, n_neuronios: int):
        super().__init__(n_neuronios)


class CamadaEscondida(CamadaMLP):
    def __init__(self, n_neuronios: int):
        super().__init__(n_neuronios)

    def calcula_correcao(self, valores_in: list, entradas: list, termos_inf_erro_saida: list):
        """ Calcula o erro e as devidas correções dos pesos dos neuronios da camada escondida """
        correcoes = {}
        for indice, (neuronio, valor_in, termo_inf_erro_saida) in enumerate(zip(self.neuronios, valores_in, termos_inf_erro_saida)):
            termo_inf_erro_saida = sum(termo_inf_erro_saida) if isinstance(
                termo_inf_erro_saida, list) else termo_inf_erro_saida
            termo_inf_erro = sum(termo_inf_erro_saida * neuronio.pesos[indice_peso]
                                 for indice_peso in neuronio.pesos) * neuronio.derivada_funcao_ativacao(valor_in)

            correcoes_pesos, correcao_bias = neuronio.calcula_correcao(
                termo_inf_erro, entradas)
            correcoes[indice] = correcoes_pesos, correcao_bias
        return correcoes


class CamadaSaida(CamadaMLP):
    def __init__(self, n_neuronios: int):
        super().__init__(n_neuronios)

    def calcula_correcao(self, y_real: list, y_calculado: list, valores_in: list, saidas_escondida: list):
        """ Calcula o erro e as devidas correções da camada de saída """
        erros = []
        termos_inf_erro = []
        correcoes = {}

        y_real = y_real if isinstance(y_real, list) else [y_real]

        for indice, (neuronio, y_r, y_c, valor_in) in enumerate(zip(self.neuronios, y_real, y_calculado, valores_in)):
            erro, termo_inf_erro = neuronio.calcula_erro(y_r, y_c, valor_in)
            correcoes_pesos, correcao_bias = neuronio.calcula_correcao(
                erro, saidas_escondida)
            erros.append(erro)
            termos_inf_erro.append(termo_inf_erro)
            correcoes[indice] = correcoes_pesos, correcao_bias

        return erros, correcoes, termos_inf_erro


class MultilayerPerceptron(object):
    """ Classe que representa uma rede neural multilayer perceptron 

    Parâmetros:
    n_entrada: int
        Número de neurônios na camada de entrada
    n_escondida: int
        Número de neurônios na camada escondida
    n_saida: int
        Número de neurônios na camada de saída
    tx_aprendizagem: float
        Taxa de aprendizagem utilizada no treinamento
    """

    def __init__(self, n_entrada: int, n_escondida: int, n_saida: int, tx_aprendizagem=0.01, limiar: float = 0.001, gerenciador_logs: GerenciadorLogs = None):
        self.camada_entrada = CamadaEntrada(n_entrada)
        self.camada_escondida = CamadaEscondida(n_escondida)
        self.camada_saida = CamadaSaida(n_saida)
        self.tx_aprendizagem = tx_aprendizagem
        self.limiar = limiar
        self.gerenciador_logs = gerenciador_logs

    def altera_pesos(self, camada: CamadaMLP, correcoes: list):
        """ Altera os pesos dos neuronios da camada de acordo com as correções """
        for neuronio, correcao in zip(camada.neuronios, correcoes):
            neuronio.altera_pesos(correcoes[correcao])

    def treina(self, X: list, y: list, epocas: int = 100):
        self.camada_escondida.inicializa_pesos(
            self.camada_entrada.n_neuronios, tx_aprendizagem=self.tx_aprendizagem)
        self.camada_saida.inicializa_pesos(
            self.camada_escondida.n_neuronios, tx_aprendizagem=self.tx_aprendizagem)

        for epoca in range(epocas):
            erro_geral = 0

            for x, y_ in zip(X, y):
                """Forward"""
                # Calcula a saída da camada escondida
                z_ins, saidas_escondida = self.camada_escondida.calcula_saida(
                    x)

                # Calcula a saída da camada de saída
                y_ins, saidas_saida = self.camada_saida.calcula_saida(
                    saidas_escondida)

                """Backpropagation"""
                # Calcula o erro e correção da camada de saída
                erro_saida, correcao_saida, termos_inf_erro_saida = self.camada_saida.calcula_correcao(
                    y_real=y_, y_calculado=saidas_saida, valores_in=y_ins, saidas_escondida=saidas_escondida)

                # Calcula o erro e a e correção da camada escondida
                correcao_escondida = self.camada_escondida.calcula_correcao(
                    valores_in=z_ins, entradas=x, termos_inf_erro_saida=termos_inf_erro_saida)

                # Faz a alteração dos pesos da camada de saída e da camada escondida
                self.altera_pesos(self.camada_saida, correcao_saida)
                self.altera_pesos(self.camada_escondida, correcao_escondida)

                erro_inter = 0
                for erros in erro_saida:
                    erro_inter = [err ** 2 for err in erros]
                    erro = sum(erro_inter)
                    erro_geral = erro_geral + erro

            erro_geral = erro_geral / len(X)

            self.gerenciador_logs.adiciona_log(self.__class__.__name__, x,  saidas_saida, self.camada_escondida.neuronios,
                                               self.camada_saida.neuronios, self.tx_aprendizagem, epoca, -1, self.limiar, epocas, erro, erro_geral)

            print(f"Epoca: {epoca} - Erro: {erro_geral}")

            if erro_geral < self.limiar:
                break

    def prediz(self, X: list):
        predito = []

        for x in X:
            _, saidas_escondida = self.camada_escondida.calcula_saida(x)
            _, saidas_saida = self.camada_saida.calcula_saida(saidas_escondida)
            predito.append(saidas_saida)

        return predito

    def to_json(self):
        """Função que desserializa a rede neural e transforma em um objeto json"""
        modelo = {
            'tx_aprendizagem': self.tx_aprendizagem,
            'camada_entrada':  self.camada_entrada.n_neuronios,
            'camada_escondida': {
                'n_neuronios': self.camada_escondida.n_neuronios,
                'neuronios': [
                    {
                        'pesos': neuronio.pesos,
                        'bias': neuronio.bias
                    } for neuronio in self.camada_escondida.neuronios

                ]
            },
            'camada_saida': {
                'n_neuronios': self.camada_saida.n_neuronios,
                'neuronios': [
                    {
                        'pesos': neuronio.pesos,
                        'bias': neuronio.bias
                    } for neuronio in self.camada_saida.neuronios
                ]
            }
        }
        return modelo

    def from_json(modelo):
        """Função que serializa a rede neural e transforma em um objeto python"""
        mlp = MultilayerPerceptron(
            modelo['camada_entrada'],
            modelo['camada_escondida']['n_neuronios'],
            modelo['camada_saida']['n_neuronios'],
            modelo['tx_aprendizagem']
        )
        for neuronio in modelo['camada_escondida']['neuronios']:
            mlp.camada_escondida.neuronios.append(
                Perceptron(neuronio['pesos'], neuronio['bias'], mlp.tx_aprendizagem))
        for neuronio in modelo['camada_saida']['neuronios']:
            mlp.camada_saida.neuronios.append(
                Perceptron(neuronio['pesos'], neuronio['bias'], mlp.tx_aprendizagem))
        return mlp
