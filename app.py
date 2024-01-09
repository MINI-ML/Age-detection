import tkinter as tk
from tkinter import filedialog
import cv2 
from PIL import Image, ImageTk
import os
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


class WebcamApp:
    def __init__(self, canvas):
        self.canvas = canvas
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
        self.canvas.delete("all")

    def update_webcam_face(self):
        if self.camera == False:
            return
        ret, frame = self.video_capture.read()
        if ret:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            self.current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(image=self.current_image)
            self.canvas.create_image(0,0, image=self.photo, anchor=tk.NW)
            window.after(15, self.update_webcam_face)
    
    def download_image(self):
        if self.current_image is not None:
            file_path = os.path.expanduser("~/webcam/captured_image.jpg")
            self.current_image.save(file_path)

class MediaApp:
    def __init__(self, canvas):
        load_video_button.config(command=self.load_video)
        load_image_button.config(command=self.load_image)
        self.canvas = canvas

    def load_image(self):
        image_path = filedialog.askopenfilename(
            title="Select a file",
        )
        image = cv2.imread(image_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        output_path = 'image_with_faces.jpg'
        cv2.imwrite(output_path, image)

        self.current_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(image=self.current_image)
        self.canvas.create_image(0,0, image=self.photo, anchor=tk.NW)

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
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            self.out.write(frame)

            self.current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(image=self.current_image)
            self.canvas.create_image(0,0, image=self.photo, anchor=tk.NW)

            window.after(int(1000 / self.cap.get(cv2.CAP_PROP_FPS)), self.play_video)
        else:
            self.cap.release()
            self.out.release()
            self.canvas.delete("all")


if __name__ == "__main__":
    window = tk.Tk()
    window.title("Webcam App")

    canvas = tk.Canvas(window, width=1000, height=720)
    canvas.grid(row=0, column=0, rowspan=10)
    download_button = tk.Button(window, text="Capture")

    start_camera_button = tk.Button(window, text="Turn on camera")
    stop_camera_button = tk.Button(window, text="Turn off camera")

    load_video_button = tk.Button(window, text="Load video")
    load_image_button = tk.Button(window, text="Load image")
    webcam_app = WebcamApp(canvas)
    media_app = MediaApp(canvas)

    download_button.grid(row=0, column=1)
    start_camera_button.grid(row=1, column=1)
    stop_camera_button.grid(row=2, column=1)
    load_video_button.grid(row=3, column=1)
    load_image_button.grid(row=4, column=1)
    window.mainloop()