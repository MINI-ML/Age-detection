import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk,ImageDraw,ImageFont
from mtcnn import MTCNN
from keras.models import load_model
import numpy as np
from moviepy.editor import VideoFileClip

selected_image = None

def action_one():
    label.config(text="Wybrano działanie 1")
    select_image()

def select_image():
    global selected_image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        selected_image = Image.open(file_path)
        detect_face(file_path)
        label.config(text=f"Wybrano zdjęcie: {file_path}")
    else:
        label.config(text="Nie wybrano żadnego zdjęcia")
def draw_text_on_image(image, text, x, y):
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("arial.ttf", 20)  # Ładuje czcionkę Arial o rozmiarze 20

    draw.text((x, y), text, fill=(0, 255, 0),font=font)  # Ustawienie koloru tekstu na biały
    image[:] = np.array(pil_image)


detector = MTCNN()

def detect_face(file_path):
    global selected_image
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    faces = detector.detect_faces(image)
    
    if faces:
        for face in faces:
            x, y, width, height = face['box']
            cv2.rectangle(image, (x, y), (x+width, y+height), (255, 0, 0), 3)
            face_img = image[y:y+height, x:x+width]
            age = predict_age(face_img)
            draw_text_on_image(image, f"Age: {age}", x, y)


    image_rgb = Image.fromarray(image)
    selected_image = image_rgb
    display_image(selected_image)
from keras import backend as K
def accuracy(y_true, y_pred, tolerance=5):  # Tolerancja +/- 5 lat
    return K.mean(K.abs(y_pred - y_true) < tolerance)
def predict_age(face_img):

    # Przygotowanie obrazu twarzy dla modelu
    face_img = cv2.resize(face_img, (200, 200))

    face_img = np.expand_dims(face_img, axis=0)
    #face_img = face_img.astype('float32') / 255.0

    # Prognozowanie wieku
    predicted_age = model.predict(face_img)
    return (int)(predicted_age[0][0])

def display_image(img):
    img = img.resize((400, 300))
    photo = ImageTk.PhotoImage(image=img)
    image_label.config(image=photo)
    image_label.image = photo

def on_closing():
    if messagebox.askokcancel("Zamknij aplikację", "Czy na pewno chcesz zamknąć aplikację?"):
        root.destroy()

def detect_camera():
    for i in range(10):  # Przeszukaj do 10 urządzeń
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Kamera o numerze identyfikacyjnym {i} jest dostępna")
            cap.release()
    global selected_image

    #while True:
    # ret, frame = cap.read()  # Odczytanie ramki z kamery

    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Konwersja ramki do formatu RGB
    #     image = Image.fromarray(frame_rgb)  # Konwersja ramki do formatu PIL.Image

    #     display_image(image)  # Wyświetlenie obrazu w interfejsie Tkinter

    #     # if cv2.waitKey(1) & 0xFF == ord('q'):  # Zakończenie pętli po wciśnięciu klawisza 'q'
    #     #     break

    cap.release()  # Zatrzymanie przechwytywania z kamery
    cv2.destroyAllWindows()  # Zamknięcie okna obrazu z kamery

def run_video():
    video_path = 'test_movie.mov'
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    output_path = 'output_movie_with_faces.mp4'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detected_face_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_img = frame[y:y+height, x:x+width]
            age = predict_age(face_img)
            draw_text_on_image(frame, f"Age: {age}", x, y)
            out.write(frame)
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    video_clip = VideoFileClip(output_path)
    video_clip.preview(fullscreen=False)


def update(cap):
        ret, frame = cap.read()

        if ret:
            display_image(frame.copy())

        root.after(10,update)

root = tk.Tk()
root.title("Aplikacja z wykrywaniem twarzy i prognozowaniem wieku")

from tkinter import ttk
style = ttk.Style()

# Ustawianie stylu dla przycisku
#style.configure('Custom.TButton', foreground='blue', background='lightgrey')

root.attributes('-fullscreen', True)

label = tk.Label(root, text="Wybierz działanie:")
label.pack()

button1 = tk.Button(root, text="Zdjęcie", command=action_one)
button1.pack()

button_camera = tk.Button(root, text="Kamera", command=detect_camera)
button_camera.pack()


button_camera = tk.Button(root, text="Video", command=run_video)
button_camera.pack()

image_label = tk.Label(root)
image_label.pack()

close_button = tk.Button(root, text="Zamknij", command=on_closing)
close_button.pack()
def on_escape(event):
    root.destroy()
root.bind('<Escape>', on_escape)  # Zbindowanie klawisza Escape do funkcji on_escape

root.protocol("WM_DELETE_WINDOW", on_closing)
# Wczytanie modelu Keras do prognozowania wieku
model = load_model('./reggresion_model2.keras',compile=False)  # Wymaga ścieżki do twojego wytrenowanego modelu
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
root.mainloop()
