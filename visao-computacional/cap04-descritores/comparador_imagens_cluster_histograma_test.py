'''
Neste arquivo estão os casos de teste da classe AgrupadorImagens
'''

import unittest
import os.path

from cv2 import sort
from comparador_imagens_cluster_histograma import AgrupadorImagens

class Test_agrupar(unittest.TestCase):

    def test_diretorio_valido(self):

        #Cria a lista de elefantes esperada
        dir_corr = os.path.dirname(os.path.abspath(__file__))
        dir_elefantes = f'{dir_corr}/elefantes'

        agrupador = AgrupadorImagens(dir_elefantes)
        print(agrupador.agrupar(2))

if __name__ == "__main__":
    unittest.main()