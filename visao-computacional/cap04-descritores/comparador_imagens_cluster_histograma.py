'''
Este algoritmo realiza a comparação de imagens utilizando o algoritmo kMeans. Para 
isso são calculados os histogramas dos canais de cores. 
'''
from logging import raiseExceptions
import os.path
from time import time
from cv2 import kmeans
from imutils import paths
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

global_seed = 1

class AgrupadorImagens:
    def __init__(self, diretorio_imagens: str):

        self.__diretorio_imagens = diretorio_imagens
    
    @property
    def diretorio_imagens(self) -> str:
        return self.__diretorio_imagens

    def __carregar_imagens(self):

        # Lista as imagens contidas no diretório
        lista_caminhos_imagens = list(paths.list_images(self.__diretorio_imagens))

        # Lista os nomes dos arquivos de imagens
        lista_nomes_imagens = [os.path.basename(caminho) for caminho in lista_caminhos_imagens]

        # Cria um dataframe
        d = {'caminho': lista_caminhos_imagens, 'hist_normalizado': None}
        self.__df_imagens = pd.DataFrame(data = d, index = lista_nomes_imagens)

        # Para linha do dataframe, carrega a imagem, calcula o histograma normalizado
        # e armazena no Dataframe
        for i, row in self.__df_imagens.iterrows():
            # carrega a imagem
            img = Image.open(row.caminho)

            # Calcula o histograma normalizado
            hist = img.histogram()
            max_hist = max(hist)
            min_hist = min(hist)
            hist_norm = list([(h - min_hist)/(max_hist - min_hist) for h in hist])

            # Adiciona ao dataframe
            self.__df_imagens.at[i, 'hist_normalizado'] = hist_norm
    
    def agrupar(self, n_clusters:int) -> pd.DataFrame:

        if(n_clusters < 2):
            raise ValueError(f'n_cluster não pode ser menor que 2')

        self.__carregar_imagens()

        kmeans = KMeans(n_clusters=n_clusters, random_state=global_seed)
        labels = kmeans.fit_predict(list(self.__df_imagens['hist_normalizado']))

        # Adiciona os labels ao Dataframe
        self.__df_imagens.insert(2, 'rotulo', labels)
        
        return self.__df_imagens.copy()
if __name__ == "__main__":
    pass