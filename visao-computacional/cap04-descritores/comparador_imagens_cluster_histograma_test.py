'''
Neste arquivo estão os casos de teste da classe AgrupadorImagens
'''

import unittest
import os.path

from cv2 import sort
from comparador_imagens_cluster_histograma import AgrupadorImagens

class Test_agrupar(unittest.TestCase):

    def test_diretorio_elefantes(self):

        #Cria a lista de elefantes esperada
        dir_corr = os.path.dirname(os.path.abspath(__file__))
        dir_imagens = f'{dir_corr}/elefantes'

        agrupador = AgrupadorImagens(dir_imagens)
        print(f'\nAgrupamento das imagens no diretório {dir_imagens}')
        print(agrupador.agrupar(2))

    def test_diretorio_carros_elefantes(self):

        #Cria a lista de elefantes esperada
        dir_corr = os.path.dirname(os.path.abspath(__file__))
        dir_imagens = f'{dir_corr}/carros_elefantes'

        agrupador = AgrupadorImagens(dir_imagens)
        
        print(f'\nAgrupamento das imagens no diretório {dir_imagens}')
        print(agrupador.agrupar(2))

if __name__ == "__main__":
    unittest.main()