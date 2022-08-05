from PIL import Image
import os.path

dir = os.path.dirname(os.path.abspath(__file__))

im01 = Image.open(f'{dir}/elefantes/elefante01.jpg')
im01.show()

from imutils import paths

lst = list(paths.list_images(dir))
print(len(lst))

print(f"file: {__file__}")
print(f"abspath: {os.path.abspath(__file__)}")
print(f"dirname: {os.path.dirname(__file__)}")