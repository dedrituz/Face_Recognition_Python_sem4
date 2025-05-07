import tkinter as tk
from tkinter import messagebox, simpledialog, Toplevel, Label, PhotoImage
import csv
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import threading
import time

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition System")
        self.csv_file = "people.csv"
        self.haar_cascade_path = "haarcascade_frontalface_default.xml"
        self.data_folder = "data"
        self.recognition_model_path = "recognition_model.xml"
        os.makedirs(self.data_folder, exist_ok=True)
        self.vid_src = None  # Video capture object
        self.camera_canvas = None
        self.current_p_id = None
        self.current_name = None
        self.cam_width = 0
        self.cam_height = 0
        self.face_cascade = None # Initialize face cascade
        self.recognizer = None
        self.people_dict = {}
        self.create_widgets()
        self.load_face_cascade()
        self.load_trained_model()
        self.load_people_data()
        self.detection_enabled = False
        self.confidence_threshold = 50  # Lower value = stricter matching

    def load_face_cascade(self):
        try:
            self.face_cascade = cv2.CascadeClassifier(self.haar_cascade_path)
            if self.face_cascade.empty():
                messagebox.showerror("Error", f"Haar cascade file not found at: {self.haar_cascade_path}")
                self.face_cascade = None
        except Exception as e:
            messagebox.showerror("Error", f"Error loading Haar cascade: {e}")
            self.face_cascade = None

    def create_widgets(self):
        self.detect_button = tk.Button(self, text="Detect", command=self.open_detect_window)
        self.detect_button.pack(pady=10)
        self.register_button = tk.Button(self, text="Register", command=self.open_register_window)
        self.register_button.pack(pady=10)

    def get_next_person_id(self):
        try:
            with open(self.csv_file, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader, None)
                if header is None:
                    return 1
                max_id = 0
                for row in reader:
                    try:
                        p_id = int(row[0])
                        max_id = max(max_id, p_id)
                    except (ValueError, IndexError):
                        pass
                return max_id + 1
        except FileNotFoundError:
            return 1

    def open_register_window(self):
        if self.face_cascade is None:
            messagebox.showerror("Error", "Face detection not initialized. Please check the Haar cascade file.")
            return
        self.current_p_id = self.get_next_person_id()
        name = simpledialog.askstring("Register", f"Enter your name (ID: {self.current_p_id}):")
        if name:
            self.current_name = name
            self.open_camera_preview_window(is_login=False)
        elif name is not None:
            messagebox.showinfo("Registration Info", "Registration cancelled or name not provided.")

    def open_camera_preview_window(self, is_login=False):
        try:
            if self.face_cascade is None:
                messagebox.showerror("Error", "Face detection not initialized.")
                return

            self.vid_src = cv2.VideoCapture(0)
            if not self.vid_src.isOpened():
                messagebox.showerror("Error", "Could not open camera.")
                return

            self.cam_width = int(self.vid_src.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
            self.cam_height = int(self.vid_src.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2

            canvas_width = self.cam_width + 200
            canvas_height = self.cam_height + 200

            if is_login:
                self.detect_window = Toplevel(self)
                self.detect_window.title("Detect Faces")
                self.camera_canvas = tk.Canvas(self.detect_window, width=canvas_width, height=canvas_height)
                self.camera_canvas.pack()

                def close_detect_window():
                    if self.vid_src is not None:
                        self.vid_src.release()
                    if hasattr(self, 'detect_window') and self.detect_window.winfo_exists():
                        self.detect_window.destroy()
                    self.detection_enabled = False

                self.detect_window.protocol("WM_DELETE_WINDOW", close_detect_window)

                def update_frame():
                    if self.vid_src is not None and self.vid_src.isOpened():
                        ret, frame = self.vid_src.read()
                        if ret:
                            mirrored_frame = cv2.flip(frame, 1)
                            resized_frame = cv2.resize(mirrored_frame, (self.cam_width, self.cam_height))
                            frame_with_faces = self.detect_faces(resized_frame, is_login) # Pass is_login
                            frame_rgb = cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(frame_rgb)
                            imgtk = ImageTk.PhotoImage(image=img)
                            self.camera_canvas.imgtk = imgtk
                            self.camera_canvas.create_image((canvas_width // 2), ((canvas_height // 4) + 100), image=imgtk, anchor=tk.CENTER)
                    if is_login:
                        self.detect_window.after(10, update_frame)
                    else:
                        self.preview_window.after(10, update_frame)
                update_frame()
                if is_login:
                    self.detection_enabled = True

            else:
                self.preview_window = Toplevel(self)
                self.preview_window.title(f"Camera Preview for {self.current_name} (ID: {self.current_p_id})")
                self.camera_canvas = tk.Canvas(self.preview_window, width=canvas_width, height=canvas_height)
                self.camera_canvas.pack()
                self.train_button = tk.Button(self.preview_window, text="Start Training", command=self.start_training)
                self.train_button.pack(pady=10)

                def close_preview_window():
                    if self.vid_src is not None:
                        self.vid_src.release()
                    if hasattr(self, 'preview_window') and self.preview_window.winfo_exists():
                        self.preview_window.destroy()
                self.preview_window.protocol("WM_DELETE_WINDOW", close_preview_window)

                def update_frame():
                    if self.vid_src is not None and self.vid_src.isOpened():
                        ret, frame = self.vid_src.read()
                        if ret:
                            mirrored_frame = cv2.flip(frame, 1)
                            resized_frame = cv2.resize(mirrored_frame, (self.cam_width, self.cam_height))
                            frame_with_faces = self.detect_faces(resized_frame, is_login) # Pass is_login
                            frame_rgb = cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(frame_rgb)
                            imgtk = ImageTk.PhotoImage(image=img)
                            self.camera_canvas.imgtk = imgtk
                            self.camera_canvas.create_image((canvas_width // 2), ((canvas_height // 4) + 100), image=imgtk, anchor=tk.CENTER)
                    self.preview_window.after(10, update_frame)
                update_frame()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def start_training(self):
        if self.vid_src is not None and self.vid_src.isOpened() and self.current_p_id is not None and self.current_name is not None:
            if hasattr(self, 'preview_window') and self.preview_window.winfo_exists():
                self.preview_window.destroy()
            self.capture_training_images()
        else:
            messagebox.showerror("Error", "Camera not initialized or user data missing.")


    def capture_training_images(self):
        try:
            if self.face_cascade is None:
                messagebox.showerror("Error", "Face detection not initialized.")
                return

            training_window = Toplevel(self)
            training_window.title(f"Capturing Images for {self.current_name} (ID: {self.current_p_id})")

            progress_label = Label(training_window, text="Capturing images: 0 / 200")
            progress_label.pack(pady=10)

            image_canvas_width = 100
            image_canvas_height = 100
            captured_images_canvas = tk.Canvas(training_window, width=5 * image_canvas_width, height=4 * image_canvas_height)
            captured_images_canvas.pack()
            captured_image_refs = []

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open camera for training.")
                return

            sample_count = 0
            max_samples = 200
            capture_interval = 50

            def capture_frame():
                nonlocal sample_count
                ret, frame = cap.read()
                if ret:
                    mirrored_frame = cv2.flip(frame, 1)
                    resized_frame = cv2.resize(mirrored_frame, (self.cam_width, self.cam_height))
                    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30)
                    )

                    face_to_crop = None
                    if len(faces) > 0:
                        largest_face = max(faces, key=lambda f: f[2] * f[3])
                        x, y, w, h = largest_face
                        face_to_crop = gray[y:y + h, x:x + w]

                    if face_to_crop is not None and face_to_crop.size > 0 and sample_count < max_samples:
                        filename = os.path.join(self.data_folder, f"{self.current_p_id}.{sample_count + 1}.jpg")
                        cv2.imwrite(filename, face_to_crop)

                        progress_label.config(text=f"Capturing images: {sample_count + 1} / {max_samples}")

                        resized_face = cv2.resize(face_to_crop, (image_canvas_width, image_canvas_height))
                        img = Image.fromarray(resized_face)
                        imgtk = ImageTk.PhotoImage(image=img)
                        captured_image_refs.append(imgtk)
                        row = sample_count // 5
                        col = sample_count % 5
                        captured_images_canvas.create_image(col * image_canvas_width, row * image_canvas_height, anchor=tk.NW, image=imgtk)

                        sample_count += 1

                if sample_count < max_samples:
                    training_window.after(capture_interval, capture_frame)
                else:
                    progress_label.config(text="Capturing complete. Training model...")
                    cap.release()
                    self.train_model()
                    self.add_person_to_csv(self.current_p_id, self.current_name)
                    training_window.destroy()

            capture_frame()

            def close_training_window():
                nonlocal sample_count
                if cap is not None:
                    cap.release()
                if training_window.winfo_exists() and sample_count < max_samples and sample_count > 0:
                    messagebox.showinfo("Warning", "Image capture incomplete. Model might not be fully trained.")
                if training_window.winfo_exists():
                    training_window.destroy()

            training_window.protocol("WM_DELETE_WINDOW", close_training_window)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            if hasattr(self, 'cap') and cap is not None:
                cap.release()

    def train_model(self):
        faces = []
        ids = []

        for root, dirs, files in os.walk(self.data_folder):
            for file in files:
                if file.endswith(".jpg"):
                    path = os.path.join(root, file)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        id_str = file.split(".")[0]
                        try:
                            person_id = int(id_str)
                            faces.append(img)
                            ids.append(person_id)
                        except ValueError:
                            print(f"Warning: Could not parse person ID from filename: {file}")

        if faces:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.train(faces, np.array(ids))
            self.recognizer.write(self.recognition_model_path)
            messagebox.showinfo("Training Info", "Face recognition model trained and saved.")
        else:
            messagebox.showinfo("Training Info", "No face data available for training.")

    def add_person_to_csv(self, p_id, name):
        try:
            with open(self.csv_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([p_id, name])
        except Exception as e:
            messagebox.showerror("Error", f"Error writing to CSV: {e}")

    def load_people_data(self):
        try:
            with open(self.csv_file, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader, None)
                if header:
                    for row in reader:
                        try:
                            p_id = int(row[0])
                            name = row[1]
                            self.people_dict[p_id] = name
                        except (ValueError, IndexError):
                            print(f"Warning: Skipping invalid row in CSV: {row}")
        except FileNotFoundError:
            messagebox.showinfo("Info", f"File '{self.csv_file}' not found.  No existing user data loaded.")
            self.people_dict = {}

    def load_trained_model(self):
        try:
            if os.path.exists(self.recognition_model_path):
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.recognizer.read(self.recognition_model_path)
            else:
                messagebox.showinfo("Info", "No existing model found.  New model will be trained on registration.")
                self.recognizer = None
        except Exception as e:
            messagebox.showerror("Error", f"Error loading trained model: {e}")
            self.recognizer = None

    def open_detect_window(self):
        try:
            with open(self.csv_file, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)
                if not any(row for row in reader):
                    messagebox.showinfo("Detect Info", "No known faces available. Please register first.")
                else:
                    self.open_camera_preview_window(is_login=True)
        except FileNotFoundError:
            messagebox.showerror("Error", f"The file '{self.csv_file}' was not found.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def detect_faces(self, frame, is_login=False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        names = []

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Draw rectangle
            face_roi = gray[y:y + h, x:x + w]
            if self.recognizer is not None:
                label_id, confidence = self.recognizer.predict(face_roi)
                if confidence < self.confidence_threshold:
                    if label_id in self.people_dict:
                        name = self.people_dict[label_id]
                        cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # Display name and confidence
                    else:
                         cv2.putText(frame, f"UNKNOWN ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, f"UNKNOWN ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "UNKNOWN", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return frame

    def close_detect_window(self):
        if hasattr(self, 'detect_window') and self.detect_window.winfo_exists():
            if hasattr(self, 'vid_src') and self.vid_src is not None:
                self.vid_src.release()
            self.detect_window.destroy()
        self.detection_enabled = False

    def start_detection(self):
        self.detection_enabled = True

if __name__ == "__main__":
    app = Application()
    app.mainloop()
