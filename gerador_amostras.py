from itertools import cycle, islice
from random import randint

from numpy import array_equal

class Amostra:
    def __init__(self, folds, holdout = True):
        self.folds = [self.desagrupando_holdout(folds)] if holdout is True else [ self.desagrupando_k_fold(fold) for fold in folds ]

    def obter_treinamento(self, indice):
        return self.folds[indice][0]

    def obter_validacao(self, indice):
        return self.folds[indice][1]
    
    def obter_fold(self, indice):
        return self.folds[indice]
    
    def desagrupando_k_fold(self, fold):
        conjunto_de_treinamento, conjunto_de_validacao = fold
        treinamento_desagrupado = self.desagrupa_conjunto(conjunto_de_treinamento)
        validacao_desagrupado = self.desagrupa_validacao(conjunto_de_validacao[0])

        return [treinamento_desagrupado, validacao_desagrupado]

    def desagrupando_holdout(self, fold):
        treinamento, validacao = fold 

        treinamento_desagrupado = self.desagrupa_conjunto(treinamento)
        validacao_desagrupado = self.desagrupa_validacao(validacao)

        return [ treinamento_desagrupado, validacao_desagrupado ]

    def desagrupa_validacao(self, validacao):
        conjunto_de_X = []
        conjunto_de_y = []

        Xs, ys = list(zip(*validacao))

        conjunto_de_X.extend(Xs)
        conjunto_de_y.extend(ys)

        return [ conjunto_de_X, conjunto_de_y ]
   
    def desagrupa_conjunto(self, conjunto):
        conjunto_de_X = []
        conjunto_de_y = []

        for subgrupo in conjunto:
            Xs, Ys = list(zip(*subgrupo))
            conjunto_de_X.extend(Xs)
            conjunto_de_y.extend(Ys)

        return [ conjunto_de_X, conjunto_de_y ]


class GeradorDeAmostras:

    letras = {
        "A": [1, 0, 0, 0, 0, 0, 0],
        "B": [0, 1, 0, 0, 0, 0, 0],
        "C": [0, 0, 1, 0, 0, 0, 0],
        "D": [0, 0, 0, 1, 0, 0, 0],
        "E": [0, 0, 0, 0, 1, 0, 0],
        "J": [0, 0, 0, 0, 0, 1, 0],
        "K": [0, 0, 0, 0, 0, 0, 1],
    }

    def obtem_letra(self, alvo):
        for letra, representacao_em_lista in self.letras.items():
            if array_equal(alvo, representacao_em_lista):
                return letra

    def holdout(self, X, y):
        """"
            Recebe um dataset e separa 2/3 dos elementos para treinamento e 1/3 para validação
        """

        # Agrupamos o dataset em 7 conjuntos de 3 letras
        dataset_treinamento = self.dataset_agrupado_por_letra(X, y)

        # Pega o indice máximo que pode-se retirar elementos do conjunto
        indice_maximo = len(dataset_treinamento[0]) - 1

        # Irá retirar 1 elemento de cada conjunto e jogar para a validação
        validacao = [ letra.pop(randint(0, indice_maximo) ) for letra in dataset_treinamento]

        holdout = [ dataset_treinamento, validacao ]

        return Amostra(holdout)

    def k_fold(self, X, y):
        """"
            Recebe um dataset e gera tem 3 folds, cada fold tem:
                - 2/3 para treinamento
                - 1/3 para validacao
        """
        # Agrupamos o dataset em 7 conjuntos de 3 letras
        dataset_agrupado = self.dataset_agrupado_por_letra(X, y)

        # Transformamos o dataset em 3 conjuntos de 7 letras
        dataset_com_letras_mixadas = list(zip(*dataset_agrupado))

        # Geramos os 3 folds
        folds = self.gerar_k_folds(dataset_com_letras_mixadas)        

        return Amostra(folds, holdout=False)

    def gerar_k_folds(self, dataset):
        tamanho = len(dataset)
        folds = []

        # Iremos gerar um K-Fold de (K = 3)
        # De forma que:
        # Primeiro Fold: [1, 2] treinamento, 3 validação
        # Segundo Fold:  [2, 3] treinamento, 1 validação
        # Terceiro Fold: [3, 1] treinamento, 2 validação
        for indice in range(tamanho):
            dataset_em_ciclos = cycle(dataset)
            
            # Pega 2/3 do dataset para treinamento
            treinamento = list(islice(dataset_em_ciclos, indice, indice + 2))

            # Pega 1/3 do dataset para validacao
            validacao = list(islice(dataset_em_ciclos, 0, 1))

            folds.append([ treinamento, validacao ])
        
        return folds        


    def dataset_agrupado_por_letra(self, X, y):
        dataset_X_y = list(zip(X, y))
        dataset_agrupado = list(self.agrupar_por_letras(dataset_X_y).values())

        return dataset_agrupado

    def agrupar_por_letras(self, dataset):
        grupos_de_letras = {}
        
        # Para cada elemento do dataset
        for x, letra_como_vetor in dataset:

            # Iremos obter a letra corresponde aos 7 valores
            letra = self.obtem_letra(letra_como_vetor)
            
            # Iremos pegar o conjunto a qual a letra pertence
            grupo_da_letra = grupos_de_letras.get(letra)

            # Se o conjunto for vazio, então iremos criar o conjunto com a letra
            if grupo_da_letra == None:
                grupos_de_letras[letra] = [ [x, letra_como_vetor] ]
            # Caso não seja vazio, iremos adicionar ao conjunto
            else:
                grupo_da_letra.append([ x, letra_como_vetor ])

        return grupos_de_letras