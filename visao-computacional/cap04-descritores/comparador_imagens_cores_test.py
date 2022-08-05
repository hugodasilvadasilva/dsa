'''
This block of code implements the testcase of Tic Tac Toe Board.
'''
import logging
import unittest
import setlog
from comparador_imagens_cores import ComparadorImagens as Comparador

class Test_carregar_imagens(unittest.TestCase):

    def test_diretorio_valido(self):
        setlog.set_log(logging.DEBUG)

        comparador = Comparador.carregar_imagens('elevantes/')

if __name__ == "__main__":
    unittest.main()