import cv2

# Crea un objeto de captura de video para la primera cámara disponible
cap = cv2.VideoCapture(0)

# Verifica si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
else:
    # Captura continuamente imágenes de la cámara
    while True:
        # Captura un fotograma de la cámara
        ret, frame = cap.read()

        # Verifica si el fotograma se capturó correctamente
        if ret:
            # Muestra el fotograma en una ventana
            cv2.imshow('Camera', frame)

        # Espera un tiempo (en milisegundos) y verifica si se presionó la tecla 'q' para salir del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera la captura de video y cierra todas las ventanas
    cap.release()
    cv2.destroyAllWindows()
