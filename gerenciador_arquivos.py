import pandas as pd

class GerenciadorArquivos(object):
    """ Classe para gerenciar arquivos 
    
    Parâmetros:
    caminho_arq : str
        Caminho para o arquivo
    """
    def __init__(self, caminho_arq: str):
        self.caminho_arq = caminho_arq

    def le_csv(self, nomes_cols: list = None, salvar: bool = True, df_caracteres: bool = False):
        """ Lê um arquivo CSV e retorna um dataframe do pandas formatado
        
        Parâmetros:
        nomes_cols : list
            Lista com os nomes das colunas
        salvar : bool
            Boolean que indica se o dataframe deve ser salvo ou não
        df_caracteres : bool
            Boolean que indica se o dataframe é o dataframe de caracteres  
        """

        try:
            df = pd.read_csv(self.caminho_arq)

            if df_caracteres:
                tam_y = 7
                tam_X = len(df.columns) - tam_y

                cols_x = [f"atributo{num}" for num in range(tam_X)]
                cols_y = [f"rotulo{num}" for num in range(tam_y)]
                cols = [*cols_x, *cols_y]

                df.columns = cols
            else:
                df.columns = nomes_cols if nomes_cols else df.columns

            if salvar:
                self.salva_df_excel(df, self.caminho_arq)
            
            return df
        except Exception as e:
            print(f"Falha ao ler arquivo! | Exceção: {e}")
            return None

    
    def salva_df_excel(self, df: pd.DataFrame, path: str):
        """ Salva um dataframe no formato de arquivo do Excel

        Parâmetros:
        df : pd.DataFrame
            Dataframe a ser salvo
        path : str
            Caminho para salvar o arquivo
        """

        try:
            path = path.replace(".csv", ".xlsx")
            df.to_excel(path, index=False)
        except Exception as e:
            print(f"Falha ao salvar dataframe! | Exceção: {e}")
            return None
    
    def extrai_X_y(self, df: pd.DataFrame, df_caracteres: bool = False):
        """ Extrai as colunas de treino(X) e a coluna target (y) de um dataframe

        Parâmetros:
        df : pd.DataFrame
            Dataframe a ser extraído
        df_caracteres : bool
            Boolean que indica se o dataframe é o dataframe de caracteres
        """
        try:
            if df_caracteres:  # se for um dataframe de caracteres, as últimas 7 são a coluna target
                X = df.iloc[:, :-7]
                y = df.iloc[:, -7:]
            else:
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
            
            return X, y
        except Exception as e:
            print(f"Falha ao extrair X e y! | Exceção: {e}")
            return None, None