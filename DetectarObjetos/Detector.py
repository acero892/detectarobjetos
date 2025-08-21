import cv2
import numpy as np

# --- 1. CONFIGURACIÓN ---
# Rutas a los archivos de YOLOv3
cfg_path = 'yolov3.cfg'
weights_path = 'yolov3.weights'

# Archivo con la lista de clases del COCO dataset
classes_path = 'coco.names'

# Descargar el archivo 'coco.names' de GitHub
# Contiene las 80 clases del modelo YOLO
# https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

# --- 2. CARGA DEL MODELO ---
print("[INFO] Cargando el modelo YOLOv3...")
try:
    net = cv2.dnn.readNet(weights_path, cfg_path)
    print("[INFO] Modelo YOLOv3 cargado exitosamente.")
except cv2.error as e:
    print(f"[ERROR] No se pudo cargar el modelo. Verifica que los archivos están en la carpeta correcta.")
    print(f"Detalles del error: {e}")
    exit()

# Cargar las clases de objetos desde el archivo .names
try:
    with open(classes_path, 'r') as f:
        CLASSES = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print("[ERROR] No se encontró el archivo 'coco.names'. Descárgalo y colócalo en la misma carpeta.")
    exit()



video_source = 0  # Usa '0' para la cámara web o la ruta de un archivo de video local
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"[ERROR] No se pudo abrir la fuente de video: '{video_source}'. Verifica la cámara o la ruta del archivo.")
    exit()

# --- 4. BUCLE DE DETECCIÓN ---
print("[INFO] Iniciando la detección de objetos con YOLOv3...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Fin del video. Saliendo...")
        break

    (h, w) = frame.shape[:2]

    # Prepara la imagen para el modelo:
    # El modelo YOLO espera una imagen con un tamaño específico (416x416)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (256, 256), swapRB=True, crop=False)

    # Establece la entrada y obtiene las detecciones
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)

    # Listas para almacenar las cajas, confianzas y IDs de clase
    boxes = []
    confidences = []
    classIDs = []

    # Procesa las detecciones
    for out in detections:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.5:
                # Escala las coordenadas de la caja
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                # Calcula la esquina superior izquierda
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Aplica supresión de no-máximos para eliminar cajas redundantes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Dibuja las cajas de detección finales
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            label = f"{CLASSES[classIDs[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Muestra el fotograma
    cv2.imshow("Detector de Objetos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 5. LIBERACIÓN DE RECURSOS ---
cap.release()
cv2.destroyAllWindows()
print("[INFO] Proceso finalizado.")