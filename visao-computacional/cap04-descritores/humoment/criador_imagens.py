# Gerador de imgens de 3 tipos diferentes: triganulo, retangulo/quadrado e circulo

# Para gerar 10 círculos no diretório 'imagens' com o prefixo 'c_' execute o 
# comando abaixo:

# $ python3 criador_imagens.py -d 'imagens' -q 10 -t 'circulo' -p 'c_'

# Para gerar 5 relangulos no diretório 'imagens' com o prefixo 'r_' execute o 
# comando abaixo

# $ python3 criador_imagens.py -d 'imagens' -q 5 -t 'retangulo' -p 'r_'

# recebe os argumentos
import argparse
import logging
import cv2
import numpy as np
import os
from setlog import set_log

DIMENSOES_IMAGEM = (500, 500)
BRANCO = (255,255,255)
PRETO = (0,0,0)
DIR_CORRENTE = os.path.dirname(os.path.abspath(__file__))

def gerar_circulos(diretorio: str, prefixo: str, qtd: int):

    DIR_IMAGENS = f'{DIR_CORRENTE}/{diretorio}'

    if not(os.path.exists(DIR_IMAGENS)):
        os.mkdir(DIR_IMAGENS)
    
    for i in range(qtd):
        # Gera o nome do arquvo
        nome_arquivo = f'{prefixo}{str(i).zfill(3)}.jpg'
        logging.debug(f'Criando imagem {nome_arquivo}')

        # Cria a matriz de uma imagem de 500 por 500 onde cada posição tem 3 valor (RGB)
        imagem = np.zeros((DIMENSOES_IMAGEM[0], DIMENSOES_IMAGEM[1], 3), dtype=np.int8)
        logging.debug(f'Matriz da imagem = {np.shape(imagem)}')

        # Calcula as dimensões do círculo
        raio = np.random.uniform(low=10, high=DIMENSOES_IMAGEM[0]/2, size=1).astype('int')[0]
        logging.debug(f'raio={raio}')

        # Calcula o centro do círculo
        (x, y) = np.random.uniform(low=raio, high=DIMENSOES_IMAGEM[0] - raio, size=2).astype('int')
        logging.debug(f'Centro do círulo={(x,y)}')

        # Gera o cículo na imagem
        cv2.circle(img=imagem, center=(x, y),radius=raio, color=BRANCO, thickness=-1)

        # Grava a imagem em arquivo
        caminho = f'{DIR_CORRENTE}/{diretorio}/{nome_arquivo}'
        logging.debug(f'Gerando arquivo em {caminho}')

        gerada = cv2.imwrite(filename=caminho, img=imagem)
        logging.debug(f'Imagem gerada = {gerada}')

def gerar_retangulos(diretorio: str, prefixo: str, qtd: int):

    DIR_IMAGENS = f'{DIR_CORRENTE}/{diretorio}'

    if not(os.path.exists(DIR_IMAGENS)):
        os.mkdir(DIR_IMAGENS)
    
    for i in range(qtd):
        # Gera o nome do arquvo
        nome_arquivo = f'{prefixo}{str(i).zfill(3)}.jpg'
        logging.debug(f'Criando imagem {nome_arquivo}')

        # Cria a matriz de uma imagem de 500 por 500 onde cada posição tem 3 valor (RGB)
        imagem = np.zeros((DIMENSOES_IMAGEM[0], DIMENSOES_IMAGEM[1], 3), dtype=np.int8)
        logging.debug(f'Matriz da imagem = {np.shape(imagem)}')

        # Calcula as dimensões do retângulo
        largura, altura = np.random.uniform(low=10, high=DIMENSOES_IMAGEM[0]-10, size=2).astype('int')
        logging.debug(f'Lagura={largura}, Altura={altura}')

        # Calcula a posição superior esquerda do retângulo
        (x, y) = np.random.uniform(low=0, high=DIMENSOES_IMAGEM[0] - max([altura, largura]), size=2).astype('int')
        logging.debug(f'Posição superior esquerda={(x,y)}')

        # Gera o reltângulo
        cv2.rectangle(img=imagem, pt1=(x,y), pt2=(x+largura, y + altura), color=BRANCO, thickness=-1)

        # Grava a imagem em arquivo
        caminho = f'{DIR_CORRENTE}/{diretorio}/{nome_arquivo}'
        logging.debug(f'Gerando arquivo em {caminho}')

        gerada = cv2.imwrite(filename=caminho, img=imagem)
        logging.debug(f'Imagem gerada = {gerada}')

if __name__ == "__main__":

    set_log(level=logging.DEBUG)

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--diretorio", required=True, help="diretório onde as imagens criadas devem ser armazenadas")
    ap.add_argument("-q", "--quantidade", type=int, default=10, help="quantidade de imagens a serem criadas")
    ap.add_argument("-t", "--tipo", required=True, default='circulo', help="Tipo de imagem a ser criada. Opções são: 'circulo', 'retangulo'")
    ap.add_argument("-p", "--prefixo", default="img_", help="Prefixo utilizado no nome da imagem.")
    args = vars(ap.parse_args())

    if args['tipo'] == 'circulo':
        gerar_circulos(diretorio=args['diretorio'], prefixo=args['prefixo'], qtd=args['quantidade'])
    elif args['tipo'] == 'retangulo':
        gerar_retangulos(diretorio=args['diretorio'], prefixo=args['prefixo'], qtd=args['quantidade'])
    