from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import os
import glob
import cv2

from keras.models import load_model
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class WebcamApp:
    def __init__(self):
        self.camera = False
        self.video_capture = cv2.VideoCapture(0)
        self.current_image = None

        download_button.config(command=self.download_image)
        start_camera_button.config(command=self.start_camera)
        stop_camera_button.config(command=self.stop_camera)

    def start_camera(self):
        self.camera = True
        self.current_image = None

        self.update_webcam_face()

    def stop_camera(self):
        self.camera = False
        blank_image = Image.new('RGB', (100, 100), 'white')

        photo = ImageTk.PhotoImage(blank_image)

        global image_label
        image_label.image = photo

    def update_webcam_face(self):
        if self.camera == False:
            return
        ret, frame = self.video_capture.read()
        if ret:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                face_img = frame[y:y+h, x:x+w]
                age = predict_age(face_img)
                # print(age)
                label = f"Age: {age}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_position = (x, y - 10)
                font_scale = 0.5
                font_color = (255, 255, 255)
                cv2.putText(frame, label, text_position, font, font_scale, font_color, 1)

            self.current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            global original_image
            original_image = self.current_image
            window_frame.event_generate("<Configure>")

            root.after(15, self.update_webcam_face)

    def download_image(self):
        if self.current_image is not None:
            file_path = os.path.expanduser("~/webcam/captured_image.jpg")
            self.current_image.save(file_path)

class MediaApp:
    def __init__(self):
        load_video_button.config(command=self.load_video)
        load_image_button.config(command=self.load_image)
        directory_button.config(command=self.select_directory_and_get_image_paths)

    def analyze_image(self, image_path):
        image = cv2.imread(image_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

            face_img = image[y:y+h, x:x+w]
            age = predict_age(face_img)
            # print(age)
            label = f"Age: {age}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_position = (x, y - 10)
            font_scale = 0.5
            font_color = (255, 255, 255)
            cv2.putText(image, label, text_position, font, font_scale, font_color, 1)

        return image


    def load_image(self):
        image_path = filedialog.askopenfilename(
            title="Select a file",
        )
        image = self.analyze_image(image_path)

        output_path = 'image_with_faces.jpg'
        cv2.imwrite(output_path, image)

        self.current_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        global original_image
        original_image = self.current_image
        window_frame.event_generate("<Configure>")


    def load_video(self):
        video_path = filedialog.askopenfilename(
            title="Select a file",
        )
        self.cap = cv2.VideoCapture(video_path)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = 'output_movie_with_faces.mp4'
        self.out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.play_video()

    def play_video(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                face_img = frame[y:y+h, x:x+w]
                age = predict_age(face_img)
                # print(age)
                label = f"Age: {age}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_position = (x, y - 10)
                font_scale = 0.5
                font_color = (255, 255, 255)
                cv2.putText(frame, label, text_position, font, font_scale, font_color, 1)

            self.out.write(frame)

            self.current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            global original_image
            original_image = self.current_image
            window_frame.event_generate("<Configure>")
            root.after(int(1000 / self.cap.get(cv2.CAP_PROP_FPS)), self.play_video)
        else:
            self.cap.release()
            self.out.release()

    def select_directory_and_get_image_paths(self):

        directory = filedialog.askdirectory()

        if not directory:
            print("No directory selected")
            return []

        analyzed_dir = os.path.join(directory, "analyzed_pictures")
        if not os.path.exists(analyzed_dir):
            os.makedirs(analyzed_dir)

        image_types = ['*.jpg', '*.jpeg', '*.png']


        image_paths = []
        for image_type in image_types:
            image_paths.extend(glob.glob(os.path.join(directory, image_type)))

        for image_path in image_paths:
            processed_image = self.analyze_image(image_path)
            save_path = os.path.join(analyzed_dir, os.path.basename(image_path))
            cv2.imwrite(save_path, processed_image)

        return image_paths


def predict_age(face_img):

    if mode == 3:
        save_path = 'face.jpeg'
        cv2.imwrite(save_path, face_img)
        age = detect_gender(save_path)
        os.remove(save_path)
        return age

    face_img = cv2.resize(face_img, (100, 100))
    face_img = face_img/255.0
    data = face_img.reshape(-1, 100, 100, 3)

    predictions = model.predict(data)
    return (int)(predictions[0][0])

def detect_gender(image_path):
        analysis = DeepFace.analyze(img_path=image_path, actions=['gender'], enforce_detection=False)

        if isinstance(analysis, list):
            gender_analysis = analysis[0]
        else:
            gender_analysis = analysis

        gender = gender_analysis["gender"]
        
        face_img = cv2.imread(image_path)
        face_img = cv2.resize(face_img, (64, 64))
        face_img = face_img/255.0
        data = face_img.reshape(-1, 64, 64, 3)

        if(gender=='Woman'):
            predictions = model_woman.predict(data)
        else:
            predictions = model_man.predict(data)

        return (int)(predictions[0])

def resize_image(event):

    aspect_ratio = original_image.width / original_image.height

    frame_width = event.width
    frame_height = event.height

    frame_width = window_frame.winfo_width()
    frame_height = window_frame.winfo_height()

    if frame_width / frame_height > aspect_ratio:
        new_height = frame_height
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = frame_width
        new_height = int(new_width / aspect_ratio)

    image = original_image.resize((new_width, new_height))
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo


def update_checkboxes(selected):
    global mode
    if selected == "one":
        
        mode = 1
        model = load_model('models/stanislaw_regresja.h5',compile=False)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        twovar.set(False)
        threevar.set(False)


    elif selected == "two":
        mode = 2
        model = load_model('models/best_regression_instance_model.h5',compile=False)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        onevar.set(False)
        threevar.set(False)


    elif selected == "three":
        mode = 3
        model = load_model('models/reggresion_model.keras',compile=False)
        onevar.set(False)
        twovar.set(False)


if __name__ == "__main__":

    mode = 1

    root = Tk()
    root.minsize(800, 300)
    content = ttk.Frame(root, padding=(3,3,12,12))
    window_frame = ttk.Frame(content, borderwidth=5, relief="ridge", width=200, height=100)
    namelbl = ttk.Label(content, text="Name")
    name = ttk.Entry(content)


    onevar = BooleanVar()
    twovar = BooleanVar()
    threevar = BooleanVar()

    one = ttk.Checkbutton(content, text="Model 1", variable=onevar, onvalue=True, offvalue=False, command=lambda: update_checkboxes("one"))
    two = ttk.Checkbutton(content, text="Model 2", variable=twovar, onvalue=True, offvalue=False, command=lambda: update_checkboxes("two"))
    three = ttk.Checkbutton(content, text="Model 3 (best)", variable=threevar, onvalue=True, offvalue=False, command=lambda: update_checkboxes("three"))

    one.grid(column=0, row=11)
    two.grid(column=1, row=11)
    three.grid(column=2, row=11)

    download_button = ttk.Button(content, text="Capture")

    start_camera_button = ttk.Button(content, text="Turn on camera")
    stop_camera_button = ttk.Button(content, text="Turn off camera")

    load_video_button = ttk.Button(content , text="Load video")
    load_image_button = ttk.Button(content, text="Load image")
    directory_button = ttk.Button(content, text="Choose directory")

    content.grid(column=0, row=0, sticky=(N, S, E, W))
    window_frame.grid(column=0, row=0, columnspan=3, rowspan=11, sticky=(N, S, E, W))

    download_button.grid(column=3, row=0)
    start_camera_button.grid(column=3, row=1)
    stop_camera_button.grid(column=3, row=2)
    load_video_button.grid(column=3, row=3)
    load_image_button.grid(column=3, row=4)
    directory_button.grid(column=3, row=5)

    root.columnconfigure(0, weight=1)
    content.columnconfigure(0, weight=3)
    content.columnconfigure(1, weight=3)
    content.columnconfigure(2, weight=3)
    content.columnconfigure(3, weight=1)

    root.rowconfigure(0, weight=1)
    content.rowconfigure(1, weight=1)
    content.rowconfigure(2, weight=1)
    content.rowconfigure(3, weight=1)
    content.rowconfigure(4, weight=1)
    content.rowconfigure(5, weight=1)
    content.rowconfigure(6, weight=1)
    content.rowconfigure(7, weight=1)
    content.rowconfigure(8, weight=1)
    content.rowconfigure(9, weight=1)
    content.rowconfigure(10, weight=1)

    original_image = Image.open("image_with_faces.jpg")
    photo = ImageTk.PhotoImage(original_image)

    image_label = Label(window_frame, image=photo)
    image_label.place(x=0, y=0, relwidth=1, relheight=1)

    window_frame.bind("<Configure>", resize_image)
    
    model = load_model('models/stanislaw_regresja.h5',compile=False)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


    model_woman = load_model('models/UTK_faces_woman.h5',compile=False)
    model_woman.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    model_man = load_model('models/UTK_faces_man.h5',compile=False)
    model_man.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    webcam_app = WebcamApp()
    media_app = MediaApp()

    root.mainloop()