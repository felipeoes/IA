from gerenciador_logs import GerenciadorLogs

class GerenciadorNotebook:
    """
        Classe que pega os resultados do MLP 
        e envia para a pasta do notebook
    """

    def __init__(self, logger: GerenciadorLogs = None):
        self.caminho_notebook = "./notebook/data"
        self.logger = logger

    def salvar_relatorio_das_predicoes(self):
        pass

    def salvar_erros_treinamento(self):
        erros = list(self.logger.obter_atributo("Erro epoca"))
        self.salva_vetor_como_csv("erro_treinamento_mlp", erros)        
        pass
    
    def salvar_erros_validacao(self):
        erros = list(self.logger.obter_atributo("Erro validacao epoca"))
        self.salva_vetor_como_csv("erro_validacao_mlp", erros)        
        pass
    
    def salvar_matriz_confusao(self, matriz):
        self.salva_matriz_como_csv("matriz_confusao", matriz)
        pass

    def salvar_log_treinamento(self):
        html = self.logger.log_html()
        self.salvar_arquivo("log_mlp.html", html)

    def salvar_arquivo(self, nome_arquivo, dados):
        caminho_arquivo = f"{self.caminho_notebook}/{nome_arquivo}"
        with open(caminho_arquivo, mode="w", encoding="utf-8") as arquivo:
            arquivo.write(dados)
    
    def salva_matriz_como_csv(self, nome_arquivo: str, matriz):
        caminho_arquivo = f"{self.caminho_notebook}/{nome_arquivo}.csv"
        try:
            with open(caminho_arquivo, mode="w", encoding="utf-8") as arquivo:
                for linha in matriz:
                    linha_str = ",".join(map(str, linha))
                    arquivo.write(linha_str)
                    arquivo.write("\n")
        except Exception as error:
            print(error)
            print(f"Houve um erro ao salvar {nome_arquivo} como csv")

    def salva_vetor_como_csv(self, nome_arquivo: str, vetor: list):
        caminho_arquivo = f"{self.caminho_notebook}/{nome_arquivo}.csv"
        try:
            with open(caminho_arquivo, mode="w", encoding="utf-8") as arquivo:
                vetor_str = ",".join(map(str, vetor))
                arquivo.write(vetor_str)
        except Exception as error:
            print(error)
            print(f"Houve um erro ao salvar {nome_arquivo} como csv")
