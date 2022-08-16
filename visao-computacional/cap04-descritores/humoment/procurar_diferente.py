import logging
import os
import cv2
import criador_imagens
import setlog
import numpy as np
import argparse
import criador_imagens as ci
from PIL import Image
from imutils import paths
from sklearn.metrics.pairwise import pairwise_distances

def extrair_humoments(dir_imagens: str) -> list:
    '''
    Extrai os descritores Hu moment das imagens contidas no diretório
    passado como parâmetro.
    
    ## Parâmetros
    `dir_imagens: str` contendo o diretório relativo à este script onde as imagens
    estão.
    `return: list` contendo tuplas com o nome da imagme na posição 0 e o Momento Hu
    na posião 1.
    '''

    # Calcula o caminho completo para o diretório onde estão as imagens
    dir_comp_imagens = f'{os.path.dirname(os.path.abspath(__file__))}/{dir_imagens}'
    logging.debug(f'Diretório imagens: {dir_comp_imagens}')

    # obtém a lista de caminhos para as imagens
    lista_caminhos_imagens = list(paths.list_images(dir_comp_imagens))
    logging.debug(f'Foram encontradas {len(lista_caminhos_imagens)} imagens no diretório')

    t_img_momentohu = []
    for caminho_imagem in lista_caminhos_imagens:


        #Carrega a imagem em escala de cinza
        imagem = cv2.imread(filename=caminho_imagem, flags=cv2.IMREAD_GRAYSCALE)
        logging.debug(f'Imagem {caminho_imagem} carregada')

        # Calcula a fronteira que será utilizada para converter os pixels em preto 
        # ou branco.  procedimento só é necessários para imagens com vários tons. 
        # Como o algoritmo de gerar imagens já gera uma imagem em preto e branco, esta
        # função não seria necessária 
        fronteira = cv2.threshold(src=imagem, thresh=cv2.THRESH_OTSU,maxval=255, type=cv2.THRESH_BINARY)[1]
        logging.debug(f'Fronteira={fronteira}')
        
        #Calcula o contorno.
        # Para entender os tipos de contorno (parametro mode) sugiro este post:
        # https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18
        contornos, hierarquia = cv2.findContours(image=fronteira.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)


        # Veja que uma figura pode conter vários contornos. Assim, o contorno que nos 
        # interessa é o com a maior área ou mais externo.
        contorno = max(contornos, key=cv2.contourArea)
        logging.debug(f'Contorno defindo')

        # Marca a área de interesse (ROI - Region of Interest)
        (x, y, w, h) = cv2.boundingRect(array=contorno)
        logging.debug(f'área de interesse: x={x}, w={w}, y={y}, h={h}')

        # Recorta a imagem, deixando apenas a região de interesse (ROI)
        rdi = cv2.resize(src=fronteira[y:y+h, x:x+w], dsize=(50, 50))
        

        # Calcula o Hu moment da área de interesse
        # Uma boa referência é https://learnopencv.com/shape-matching-using-hu-moments-c-python/
        # Primeiro ele calcula os momentos centrais que servirá de parâmetro para o 
        # cálculo do HuMoment
        momentos_centrais = cv2.moments(array=rdi)

        # Agora sim, calcula o Hu Moment
        momento_hu = cv2.HuMoments(m=momentos_centrais).flatten()
        logging.debug(f'Momento Hu = {momento_hu}')

        # Adicona o Hu momento (na verdade os Hu moments, já que são algumas 
        # características que são retornadas) ao dicionário de momentos 
        nome_imagem = os.path.basename(caminho_imagem)

        t_img_momentohu.append((nome_imagem, momento_hu))
        logging.debug(f'Momento Hu da imagem {nome_imagem} adicionada à lista.')
    
    return t_img_momentohu

def obter_humoment_maisdistante(lst_momentos_hu: list) -> int:
    '''
    Encontra o Hu moment mais diferente de todas os demais. Para isso utiliza
    a função `sklearn.metrics.pairwise.pairwise_distances` para calcula as 
    distâncias entre cada momento imagem.

    ##Parâmetros
    `- lst_momentos_hu: list` contendo a lista de momentos hu das imagens.
    `- return: int` com a posição do Momento Hu mais distante dos demais.
    '''
    # Obtèm os Hu Moments
    momentos_hu = lst_momentos_hu
    logging.debug(f'Obtido {len(momentos_hu)} Momentos Hu')

    # Calcula as distâncias entre todos Hu Moments
    distancias = pairwise_distances(momentos_hu).sum(axis=1)
    logging.debug(f'Distância entre os Hu Moments')
    logging.debug(f'{distancias}')

    # Obtém a posição do imagem com a maior distância entre as demais
    pos_max = np.argmax(distancias)
    logging.debug(f'Imagem mais diferente encontrada na posição {pos_max}')

    return pos_max
        

if __name__ == "__main__":

    setlog.set_log(logging.DEBUG)

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--diretorio", required=True, help="diretório onde as imagens criadas estão/serão armazenadas")
    ap.add_argument("-qr", "--quantidade_retangulos", type=int, default=1, help="quantidade de retângulos que devem ser criadas")
    ap.add_argument("-qc", "--quantidade_circulos", type=int, default=10, help="quantidade de círculos que devem ser criadas")
    args = vars(ap.parse_args())

    # Solicita a criação dos círculos e em seguida dos retângulos
    ci.gerar_circulos(diretorio=args['diretorio'], prefixo='c_', qtd=args['quantidade_circulos'])
    ci.gerar_retangulos(diretorio=args['diretorio'], prefixo='r_', qtd=args['quantidade_retangulos'])

    # Obtém os Hu moments
    t_img_mh = extrair_humoments(dir_imagens=args['diretorio'])

    # Extrai somente os momentos hu
    momentos_hu = [val[1] for val in t_img_mh]

    # Passa os hu moments como parâmetro
    max_diferente = obter_humoment_maisdistante(momentos_hu)
    nome_max_dif = t_img_mh[max_diferente][0]

    print(f'A imagem mais diferente é {nome_max_dif}')