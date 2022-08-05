'''
Este script realiza a comparação de imagens pelos canais de cores. Ela calcula a média e o desvio padrão dos canais de cores de uma imagem
calcula a distância euclidiana entre as imagens. Aquelas com menores valores entre si, são as mais parecidas.
'''
import logging
import cv2
import os.path
import numpy as np
from imutils import paths
from scipy.spatial import distance
from PIL import Image

class ComparadorImagens:

    def comparar(subpasta: str, nome_imagem_ref: str):

        # Carrega os diretórios onde estão as figuras
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        img_dir = f"{curr_dir}/{subpasta}"
        logging.debug(f'Diretorio das imagens: {img_dir}')

        # Carrega as imagens
        dic_imagens = ComparadorImagens.carregar_imagens(img_dir)

        # Retira a imagem referência
        descritor_ref = dic_imagens.pop(nome_imagem_ref)

        # Carrega a imagem de referência
        img_ref = Image.open(f'{img_dir}/{nome_imagem_ref}')

        # Percorre as imagens calculando a distância euclidiana
        # entre cada uma delas e a imagem referência
        for nome_imagem in dic_imagens.keys():
            
            # Calcula e imprime a distância euclidina e imprime na linha de comando
            dist_euc = distance.euclidean(dic_imagens[nome_imagem], descritor_ref)
            print(f"Distância de '{nome_imagem_ref}' até '{nome_imagem}' é igual a {dist_euc}")

            # Carrega a imagem
            dir_imagem = f"{img_dir}/{nome_imagem}"
            img = Image.open(dir_imagem)
            img.show(title=f'Dist Euclidiana de {nome_imagem} = {dist_euc}', )


    def carregar_imagens(diretorio: str) -> dict:

        # Carrega a lista de caminhos para cada imagem
        lst_dir_imagens = list(paths.list_images(diretorio))
        logging.debug(f'{len(lst_dir_imagens)} carregadas')

        # Percorre uma lista de imagens
        imagens = {}
        for dir_imagem in lst_dir_imagens:
            
            # Abre a imagem
            imagem = cv2.imread(dir_imagem)
            logging.debug(f"Imagem {dir_imagem} carregada")

            # Extrai o nome do arquivo
            nome_imagem = dir_imagem[dir_imagem.rfind("/") + 1:]
            logging.debug(f"Extraído o nome da imagem {nome_imagem}")

            # Calcula a média e desvio padrão da imagem
            med_dvp = cv2.meanStdDev(imagem)
            descritores = np.concatenate(med_dvp, axis=None)            
            logging.debug(f"Descritores = {descritores}")

            # Adiciona a imagem ao dicionário de imagens
            imagens[nome_imagem] = descritores
            logging.debug(f"Imagem adicionada ao dicionário de imagens. Total de imagens = {len(imagens)}")

        return imagens    

if __name__ == "__main__":
    ComparadorImagens.comparar('elefantes', 'elefante01.jpg')