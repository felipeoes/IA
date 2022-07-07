import pandas as pd
import json


class GerenciadorLogs(object):
    """ Classe para gerenciar arquivos de log do Multilayer Perceptron"""

    def __init__(self, caminho_arq: str = None):
        self.caminho_arq = caminho_arq
        self.log = pd.DataFrame(columns=["Modelo", "Entradas", "Saídas", "Neuronios Camada Oculta", "Neuronio Camada Saída", "Taxa Aprendizado",
                                "Epocas",  "Tempo", "Tolerância Erro", "Tolerância Epocas", "Erro epoca", "Erro validacao"])

    def salva_log_csv(self):
        """ Salva o log no formato de arquivo do CSV """

        try:
            self.log.to_csv(self.caminho_arq, index=False)
        except Exception as e:
            print(f"Falha ao salvar log! | Exceção: {e}")

    def salva_log_excel(self):
        """ Salva o log no formato de arquivo do Excel """

        try:
            self.log.to_excel(self.caminho_arq, index=False)
        except Exception as e:
            print(f"Falha ao salvar log! | Exceção: {e}")

    def adiciona_log(self, modelo: str, entrada: list, saida: list, neuronios_oculta: int, neuronio_saida: int, taxa_aprendizado: float, epocas: int, tempo: float,
                     tolerancia_erro: float, tolerancia_epocas: int, erro_epoca: float, erro_validacao: float):
        """ Adiciona um log ao dataframe de log """

        try:
            self.log = pd.concat([self.log, pd.DataFrame([[modelo, entrada, saida, [neuronio.converte_json() for neuronio in neuronios_oculta], [neuronio.converte_json() for neuronio in neuronio_saida], taxa_aprendizado, epocas, tempo, tolerancia_erro, tolerancia_epocas,
                                 erro_epoca, erro_validacao]], columns=self.log.columns)], ignore_index=True)
        except Exception as e:
            print(f"Falha ao adicionar log! | Exceção: {e}")

    def obter_atributo(self, coluna: str):
        try:
            return self.log.get(coluna)
        except Exception as error:
            print(f"Houve um erro ao obter o atributo {coluna} | {error}")
            return None

    def log_completo(self):
        """ Retorna o dataframe de log completo """

        try:
            return self.log
        except Exception as e:
            print(f"Falha ao retornar dataframe de log! | Exceção: {e}")
            return None

    def log_html(self):
        """ Converte o dataframe de log para uma string no formato HTML """

        try:
            return self.log.to_html()
        except Exception as e:
            print(
                f"Falha ao converter log para string no formato HTML! | Exceção: {e}")
            return None
