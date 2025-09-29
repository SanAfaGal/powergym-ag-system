import face_recognition
import os
import pickle

# Carpeta con imágenes conocidas
known_path = "Images"

known_encodings = []
known_names = []

for filename in os.listdir(known_path):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(known_path, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])

print(f"✅ Se cargaron {len(known_encodings)} rostros conocidos.")

# Guardar en un archivo para usar luego
with open("encodings.pkl", "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print("💾 Embeddings guardados en encodings.pkl")
