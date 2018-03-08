# coding=utf-8
# Fuentes:
# - Corregir alineación: https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
# - Rellenar bordes: https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
# - Auto-cortar:
#   - https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
#   - https://stackoverflow.com/questions/37803903/opencv-and-python-for-auto-cropping

# Importar los paquetes necesarios
import numpy as np
import argparse
import cv2
 
# Construir el intérprete de argumentos, y extraer argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image file")
ap.add_argument("-o", "--out", required=True,
	help="path to output image file")
ap.add_argument("-n", "--negativeout", required=True,
	help="path to output negative image file")
args = vars(ap.parse_args())
 
imageNameOut = args["out"]
imageNameNegativeOut = args["negativeout"]

# Cargar la imagen inicial del disco
imageName = args["image"]
imageBig = cv2.imread(imageName)
image = cv2.resize(imageBig, (0,0), fx=0.6, fy=0.6) 

# Tratar de remover "ruido"
image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)

# Convertir la imagen a escala de grises, e invertir los
# colores de fondo y frente para asegurar que el frente
# sea "blanco" y el fondo "negro"
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
 
# Limitar la imagen, estableciendo todos los píxeles del
# frente a 255 (blanco total), y los de fondo a 0 (negro total)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Crear máscara para rellenar colores.
# Nótese que el tamaño debe ser 2 píxeles más grande que el de la imagen.
h, w = thresh.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

print("[INFO] Contour rectangle: {h}, {w}".format(h=h, w=w))

# Rellenar bordes externos con color negro
cv2.floodFill(thresh, mask, (0,0), 0)
cv2.floodFill(thresh, mask, (0,h-1), 0)
cv2.floodFill(thresh, mask, (w-1,0), 0)
cv2.floodFill(thresh, mask, (w-1,h-1), 0)

# Tratar de remover "ruido" otra vez
# thresh = cv2.fastNlMeansDenoising(thresh)

#cv2.imwrite("step1.png",thresh)

# Tomar las coordenadas (x, y) de todos los píxeles
# con valores mayores o iguales a cero, y usarlas para
# calcular una caja rotada que delimite todas las
# coordenadas
coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]

# Mostrar el ángulo de corrección "crudo"
print("[INFO] Raw angle: {:.3f}".format(angle))

# La función `cv2.minAreaRect` retorna valores en el rango
# [-90, 0]; como el rectángulo rota en el sentido de las
# agujas del reloj, el ángulo obtenido tiende a cero. En
# este caso especial debemos sumar 90 grados al ángulo.
if angle < -45:
	angle = -(90 + angle)
 
# De lo contrario, solo basta tomar el inverso del ángulo
# para hacerlo positivo
else:
	angle = -angle

# Rotar la imagen para alinearla
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Mostrar el ángulo de corrección final
print("[INFO] Angle: {:.3f}".format(angle))

# cv2.imwrite("step2.png",rotated)

# Convertir la nueva imagen a escala de grises, e invertir los
# colores de fondo y frente para asegurar que el frente
# sea "blanco" y el fondo "negro"
gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
 
# Limitar la imagen, estableciendo todos los píxeles del
# frente a 255 (blanco total), y los de fondo a 0 (negro total)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Crear máscara para rellenar colores.
# Nótese que el tamaño debe ser 2 píxeles más grande que el de la imagen.
h, w = thresh.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

# Rellenar bordes externos con color negro
cv2.floodFill(thresh, mask, (0,0), 0)
cv2.floodFill(thresh, mask, (0,h-1), 0)
cv2.floodFill(thresh, mask, (w-1,0), 0)
cv2.floodFill(thresh, mask, (w-1,h-1), 0)

# Encontrar posibles contornos
im2,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print("[INFO] Contours {len}".format(len = len(contours)))

# Obtener los mejores contornos
maxH, maxW = rotated.shape[:2]
best_box=[-1,-1,-1,-1]
for c in contours:
   x,y,w,h = cv2.boundingRect(c)
   print("[INFO] Contour rectangle: {x}, {y}, {w}, {h}".format(x=x, y=y,w=w, h=h))
   if best_box[0] < 0:
       best_box=[x,y,x+w,y+h]
   else:
       if x<best_box[0]:
           best_box[0]=x
       if y<best_box[1]:
           best_box[1]=y
       if x+w>best_box[2]: #and x+w < maxW-10:
           best_box[2]=x+w
       if y+h>best_box[3]: #and y+h < maxH-10:
           best_box[3]=y+h

x,y,w,h = best_box

print("[INFO] Final contour rectangle: {x}, {y}, {w}, {h}".format(x=x, y=y,w=w, h=h))

# Cortar imagen rotada, e imagen negativa rotada
crop = rotated[y:h,x:w]
cropThresh = thresh[y:h,x:w]

# Mostrar la imagen final
# cv2.imshow("Final", crop)

# Guardar imágenes

cv2.imwrite(imageNameOut,crop, [cv2.IMWRITE_PNG_COMPRESSION, 9])

#cv2.imwrite(imageNameNegativeOut,cropThresh)

#cv2.waitKey(0)
