from gerenciador_logs import GerenciadorLogs
import json

class GerenciadorNotebook:
    """
        Classe que pega os resultados do MLP 
        e envia para a pasta do notebook
    """

    def __init__(self, logger: GerenciadorLogs = None):
        self.caminho_notebook = "./notebook/data"
        self.logger = logger
        self.matriz_confusao = None
        self.predicoes = None
        self.conjuntos = None

    @property
    def conjuntos(self):
        return self._conjuntos

    @conjuntos.setter
    def conjuntos(self, conjuntos):
        self._conjuntos = conjuntos

    @property
    def matriz_confusao(self):
        return self._matriz_confusao
    
    @matriz_confusao.setter
    def matriz_confusao(self, matriz):
        self._matriz_confusao = matriz
    
    @property
    def predicoes(self):
        return self._predicoes
    
    @predicoes.setter
    def predicoes(self, predicoes):
        self._predicoes = predicoes

    def salvar_log_treinamento(self):
        html = self.logger.log_html()
        self.salvar_arquivo("log_mlp.html", html)

    def grava_notebook(self, nome_notebook = "1"):
        caminho_arquivo = f"{self.caminho_notebook}/notebook-{nome_notebook}.json"
        erros_treinamento = self.logger.obter_atributo("Erro epoca")
        erros_validacao = self.logger.obter_atributo("Erro validacao")

        with open(caminho_arquivo, mode="w", encoding="utf-8") as arquivo:
            notebook = Notebook(
                self.conjuntos,
                list(erros_treinamento),
                list(erros_validacao),
                self.matriz_confusao,
                self.predicoes
            ).to_json()

            arquivo.write(json.dumps(notebook))
    
class Notebook:
    def __init__(self, conjuntos, erros_treinamento, erros_validacao, matriz_confusao, log_predicoes):
        self.conjuntos = conjuntos
        self.erros_treinamento = erros_treinamento
        self.erros_validacao = erros_validacao
        self.matriz_confusao = matriz_confusao
        self.log_predicoes = log_predicoes

    def to_json(self):
        return {
            'conjuntos': self.conjuntos,
            'erros_treinamento': self.erros_treinamento,
            'erros_validacao': self.erros_validacao,
            'matriz_confusao': self.matriz_confusao,
            'log_predicoes': self.log_predicoes
        }