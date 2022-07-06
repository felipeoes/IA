import pandas as pd


class GerenciadorArquivos(object):
    """ Classe para gerenciar arquivos 

    Parâmetros:
    caminho_arq : str
        Caminho para o arquivo
    """

    def __init__(self, caminho_arq: str, tamanho_entrada: int = 10):
        self.caminho_arq = caminho_arq
        self.tamanho_entrada = tamanho_entrada

    def le_csv(self, nomes_cols: list = None, header=None, salvar: bool = True, df_caracteres: bool = False):
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
            df = pd.read_csv(self.caminho_arq, header=header)

            if df_caracteres:
                tam_y = self.tamanho_entrada
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
            if df_caracteres:  # se for o dataframe de caracteres, as últimas 7 são a coluna target
                y_length = self.tamanho_entrada
                X_length = len(df.columns) - y_length

                X = df.iloc[0:, 0: X_length].values.tolist()
                y = df.iloc[0:, X_length: (X_length+y_length)].values.tolist()
            else:
                X = df.loc[0: len(df.columns), :].values.tolist()
                y = []

                for col in X:
                    y.append(col.pop())

            return X, y
        except Exception as e:
            print(f"Falha ao extrair X e y! | Exceção: {e}")
            return None, None
