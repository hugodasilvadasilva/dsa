'''
# Descrição

Este script utiliza os Momentos Zernike para identificar se um objeto está 
contido dentro de uma imagem.

## Configurações do ambiente
Para que este ambiente funcione é preciso instalar os pacotes do mahotas, 
executando o comando abaixo:

$ pip install mahotas

## Utilizando o script

Para utilizar este script basta chamá-lo pela linha de comando, ex:

$ segmenta-formas.py
'''

import os
import cv2
from imutils import paths
from PIL import Image

def obter_características(caminho_imagem:str) -> list:

    #Carrega a imagem em escada de cinza
    imagem = cv2.imread(filename=caminho_imagem, flags=cv2.IMREAD_GRAYSCALE)


    # Exibe a imagem em cinza

    # Retira imperfeições/ruídos baseado nos 13 pixels vizinhos
    imagem_ajustada = cv2.GaussianBlur(src=imagem, ksize=(13, 13))

    # Converte a figura para preto e branco (type=cv2.TRESH_BINARY) utilizando um 
    # limite de 50
    valor_fronteira, imagem_peb = cv2.threshold(src=imagem_ajustada, thresh=50, maxval=255, type=cv2.THRESH_BINARY)

    #Realiza a dilatação da imagem. Esta operação faz com uma linha fina fique mais
    #espessa, ou é como se uma letra fosse convertida para negrito. Em imagens
    #colocoridas, ela aumenta o brilho da imagem.
    # Para mais detalhes, acesse: https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
    imagem_dilatada = cv2.dilate(src=imagem_peb, kernel=None, iterations=4)
    imagem_erodida = cv2.erode(src=imagem_dilatada, kernel=None, iterations=2)

    # Detecta os contornos externos (cv2.RETR_EXTERNAL)
    (_, contornos, _) = cv2.findContours(image=imagem_erodida.copy(), mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)





    
    #lista que conterá as descrições das formas
    formas = []

    # Abre a imgem em 

from PIL import Image

if __name__ == "__main__":

    dir_imagens = 'imagens'
    dir_comp_imagens = f'{os.path.dirname(os.path.abspath(__file__))}/{dir_imagens}'

    lst_caminho_imagens = list(paths.list_images(dir_comp_imagens))

    for caminho_imagem in lst_caminho_imagens:

        img = cv2.imread(filename=caminho_imagem)
        cv2.imshow('teste', img)

        imgarr = Image.fromarray(img)
        imgarr.show()
