import face_recognition
import cv2
import os

# === 1. Cargar todas las imágenes conocidas ===
known_encodings = []
known_names = []

known_path = "Images"  # Carpeta con imágenes de referencia

for filename in os.listdir(known_path):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(known_path, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            # Guardar el nombre sin extensión (ej: "mateo.jpg" -> "mateo")
            known_names.append(os.path.splitext(filename)[0])

print(f"Se cargaron {len(known_encodings)} imágenes conocidas.")

# === 2. Capturar solo una foto desde la cámara ===
video_capture = cv2.VideoCapture(0)

ret, frame = video_capture.read()
video_capture.release()

if not ret:
    print("No se pudo capturar la imagen")
    exit()

# Convertir BGR (OpenCV) → RGB (face_recognition)
rgb_frame = frame[:, :, ::-1]

# === 3. Detectar rostros en la foto ===
face_locations = face_recognition.face_locations(rgb_frame)
face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

# === 4. Comparar con las imágenes conocidas ===
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_encodings, face_encoding)
    name = "Desconocido"

    if True in matches:
        match_index = matches.index(True)
        name = known_names[match_index]

    # Dibujar rectángulo y etiqueta
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Mostrar la foto procesada
cv2.imshow("Resultado", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
