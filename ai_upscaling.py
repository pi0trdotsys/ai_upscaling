import cv2
from cv2 import dnn_superres
import os

def upscale_image_ai(input_path, output_path, model_path="EDSR_Tensorflow/models/EDSR_x4.pb"):
    if not os.path.exists(model_path):
        print(f"BŁĄD: Nie znaleziono pliku modelu '{model_path}'.")
        print("Pobierz go z: https://github.com/Saafke/EDSR_Tensorflow/tree/master/models")
        return

    print(f"Wczytywanie zdjęcia: {input_path}...")
    image = cv2.imread(input_path)
    
    sr = dnn_superres.DnnSuperResImpl_create()
    
    print("Wczytywanie modelu AI (to może chwilę potrwać)...")
    sr.readModel(model_path)
    
    sr.setModel("edsr", 4)
    
    print("Przetwarzanie obrazu (Upscaling)...")
    result = sr.upsample(image)
    
    cv2.imwrite(output_path, result)
    print(f"Gotowe! Zdjęcie zapisane jako: {output_path}")
    print(f"Nowa rozdzielczość: {result.shape[1]}x{result.shape[0]}")

moje_zdjecie = "image.jpeg" 
nazwa_wynikowa = "image_4k_ai.jpeg"

upscale_image_ai(moje_zdjecie, nazwa_wynikowa)
