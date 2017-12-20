import cv2     #llama a OpenCv
import numpy as np

#cargamos la plantilla e inicializamos la webcam:
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
#profile_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_profileface.xml')

#Captura del archivo de video
capture = cv2.VideoCapture("videos/Video1.mp4")

#Captura de video desde cámara
#capture = cv2.VideoCapture(0)

#Ventanas adicionales
#cv2.namedWindow("Video en escala de grises:", cv2.WINDOW_AUTOSIZE)

ret, frame = capture.read()

#Rotar el video
num_rows, num_cols = frame.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), 90, 1)
cos = np.abs(rotation_matrix[0, 0])
sin = np.abs(rotation_matrix[0, 1])

#compute the new bounding dimensions of the image
nW = int((num_rows * sin) + (num_cols * cos))
nH = int((num_rows * cos) + (num_cols * sin))

# adjust the rotation matrix to take into account translation
rotation_matrix[0, 2] += (nW / 2) - (num_cols / 2)
rotation_matrix[1, 2] += (nH / 2) - (num_rows / 2)

i=0

nombre = input("Ingresar nombre: ")

while(True):

    #Lectura del frame desde la señal de video (cámara o archivo de video)
    ret, frame = capture.read()

    #Si llega al final del video no habrá frame
    if(not ret):
      break

    img_rotation = cv2.warpAffine(frame, rotation_matrix, (nW, nH))

    #Convertir a escala de grises
    frameGris = cv2.cvtColor(img_rotation, cv2.COLOR_BGR2GRAY)

    #buscamos las coordenadas de los rostros (si los hay) y
    #guardamos su posicion
    faces = face_cascade.detectMultiScale(frameGris, 1.1, 5)
    #facesP = profile_cascade.detectMultiScale(img_rotation, 1.3, 5)


    # Dibujamos un rectangulo en las coordenadas de cada rostro
    for (x, y, w, h) in faces:
        cv2.rectangle(img_rotation, (x, y), (x + w, y + h), (125, 255, 0), 1)
        cropped = img_rotation[y:y + h, x:x + w]
    if i%10 == 0:
        direccion = 'Imagenes/' + nombre + repr(i) + '.jpg'
        cv2.imwrite(direccion,cropped)
    i += 1
    #for (x, y, w, h) in facesP:
    #    cv2.rectangle(img_rotation, (x, y), (x + w, y + h), (255, 125, 0), 2)

    #Muestra el video resultante en su respectiva ventana
    cv2.imshow("Video",img_rotation)
    #cv2.imshow("Video en escala de grises:",frameGris)

    key = cv2.waitKey(33)
    #Termina presionando la tecla Esc
    if (key==27):
        break

cv2.waitKey(0)
capture.release()
cv2.destroyAllWindows()