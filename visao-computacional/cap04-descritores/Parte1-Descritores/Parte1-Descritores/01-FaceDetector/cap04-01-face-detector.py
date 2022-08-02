# Detector de Faces

# Para executar essse script, use: 
# python cap04-01-face-detector.py

# Import
import cv2

# Carrega a imagem e converte para Grayscale
image = cv2.imread("royal.jpg")
cv2.imshow("Original", image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Carrega o detector de faces
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Detecta as faces
faces = detector.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 10, minSize = (30, 30))

# Loop por todas as faces e desenha um quadrado em torno delas
for (x, y, w, h) in faces:
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Mostra as faces detectadas
cv2.imshow("Faces", image)
cv2.waitKey(0)