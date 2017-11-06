import cv2     #llama a OpenCv

#cargamos la plantilla e inicializamos la webcam:
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

#Captura del archivo de video
capture = cv2.VideoCapture("videos/Video1.wmv")

#Captura de video desde cámara
#capture = cv2.VideoCapture(0)


#Presentación de las diferentes imágenes y la señal de video
cv2.namedWindow("Video:", cv2.WINDOW_AUTOSIZE)
#Ventanas adicionales
cv2.namedWindow("Video en escala de grises:", cv2.WINDOW_AUTOSIZE)

while(True):

    #Lectura del frame desde la señal de video (cámara o archivo de video)
    ret, frame = capture.read()
    #Si llega al final del video no habrá frame
    if(not ret):
      break

    #Convertir a escala de grises
    frameGris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #buscamos las coordenadas de los rostros (si los hay) y
    #guardamos su posicion
    faces = face_cascade.detectMultiScale(frameGris, 1.3, 5)

    # Dibujamos un rectangulo en las coordenadas de cada rostro
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (125, 255, 0), 2)

    #Muestra el video resultante en su respectiva ventana
    cv2.imshow("Video:",frame)
    cv2.imshow("Video en escala de grises:",frameGris)

    key = cv2.waitKey(33) #Retraso en milisegundos para leer el siguiente frame (nota para archivo de imagen poner 0 )
    #Termina presionando la tecla Esc
    if (key==27):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
