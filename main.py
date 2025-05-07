import tkinter as tk
from tkinter import messagebox, simpledialog, Toplevel, Label, Button
import csv
import cv2
from PIL import Image, ImageTk
import os
import numpy as np

class Application:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Recognition System")
        self.master.geometry("300x200")
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.p_id = None
        self.user_name = None
        self.data_dir = "data"
        self.video_capture = None
        self.is_streaming = False
        self.is_detecting = False
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.load_model()
        self.labels_dict = self.load_labels()

        self.cam_width = None
        self.cam_height = None
        self.video_canvas = None

        self.detection_params = {
            'scaleFactor': 1.2,
            'minNeighbors': 6
        }


        self.create_main_widgets()

    def load_model(self):
        if os.path.exists("recognition_model.xml"):
            self.recognizer.read("recognition_model.xml")

    def load_labels(self):
        labels = {}
        csv_path = "people.csv"
        if not os.path.exists(csv_path):
            return labels
        try:
            with open(csv_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    if len(row) >= 2 and row[0].isdigit():
                        labels[int(row[0])] = row[1]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load labels from people.csv: {e}")
        return labels

    def create_main_widgets(self):
        self.clear_master()

        self.detect_button = tk.Button(self.master, text="Detect", command=self.login, width=20)
        self.detect_button.pack(pady=(40, 10), anchor='center')

        self.register_button = tk.Button(self.master, text="Register", command=self.register_prompt, width=20)
        self.register_button.pack(pady=10, anchor='center')

        self.master.geometry("300x200")

    def clear_master(self):
        for widget in self.master.winfo_children():
            widget.destroy()

    def login(self):
        csv_path = "people.csv"
        try:
            with open(csv_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
                if len(rows) <= 1:
                    messagebox.showwarning("No Faces Found", "No known faces available. Please register first.")
                    return
        except FileNotFoundError:
            messagebox.showwarning("No Faces Found", "No known faces available. Please register first.")
            return

        self.load_model()
        self.labels_dict = self.load_labels()
        self.open_detection_window()

    def open_detection_window(self):
        self.clear_master()

        label = tk.Label(self.master, text="Face Detection", font=("Helvetica", 12))
        label.pack(pady=5)

        if self.video_capture is None or not self.video_capture.isOpened():
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                messagebox.showerror("Camera Error", "Cannot open webcam.")
                self.create_main_widgets()
                return

        frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.cam_width = frame_width // 2
        self.cam_height = frame_height // 2

        canvas_width = self.cam_width + 200
        canvas_height = self.cam_height + 200

        total_height = canvas_height + 150
        total_width = canvas_width + 20
        self.master.geometry(f"{total_width}x{total_height}")

        self.video_canvas = tk.Canvas(self.master, width=canvas_width, height=canvas_height)
        self.video_canvas.pack()

        back_button = tk.Button(self.master, text="Back", command=self.cancel_detection, width=20)
        back_button.pack(pady=10)

        self.is_detecting = True
        self.is_streaming = False
        self.update_video_stream_detect()

    def update_video_stream_detect(self):
        if not self.is_detecting or self.video_capture is None:
            return

        ret, frame = self.video_capture.read()
        if not ret:
            self.stop_video_stream()
            messagebox.showerror("Camera Error", "Failed to capture video frame.")
            return

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, **self.detection_params)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img_resized = cv2.resize(face_img, (200, 200))

            try:
                label, confidence = self.recognizer.predict(face_img_resized)
                if confidence < 35:
                    name = self.labels_dict.get(label, "UNKNOWN")
                    color = (255, 255, 255)
                    label_text = f"{name}"
                else:
                    name = "UNKNOWN"
                    color = (0, 0, 255)
                    label_text = "UNKNOWN"
            except:
                name = "UNKNOWN"
                color = (0, 0, 255)
                label_text = "UNKNOWN"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        small_frame = cv2.resize(frame, (self.cam_width, self.cam_height))
        cv2image = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)

        x_coord = (self.cam_width + 200) // 2
        y_coord = ((self.cam_height + 200) // 4) + 100

        self.video_canvas.delete("all")
        self.video_canvas.create_image(x_coord, y_coord, anchor=tk.CENTER, image=imgtk)
        self.video_canvas.imgtk = imgtk

        self.master.after(30, self.update_video_stream_detect)

    def stop_video_stream(self):
        self.is_detecting = False
        self.is_streaming = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None

    def cancel_detection(self):
        self.stop_video_stream()
        self.create_main_widgets()

    def register_prompt(self):
        csv_path = "people.csv"
        next_p_id = 1
        try:
            with open(csv_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
                if len(rows) > 1:
                    existing_ids = [int(row[0]) for row in rows[1:] if row and row[0].isdigit()]
                    if existing_ids:
                        next_p_id = max(existing_ids) + 1
        except FileNotFoundError:
            next_p_id = 1

        self.p_id = next_p_id

        prompt = f"Your assigned ID is {self.p_id}.\nPlease enter your name:"
        name = simpledialog.askstring("Register", prompt, parent=self.master)

        if name and name.strip():
            self.user_name = name.strip()
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            self.show_camera_feed()
        else:
            messagebox.showwarning("Input Error", "Name cannot be empty. Registration cancelled.")

    def show_camera_feed(self):
        self.clear_master()

        label = tk.Label(self.master, text=f"Registering {self.user_name} (ID: {self.p_id})", font=("Helvetica", 12))
        label.pack(pady=5)

        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            messagebox.showerror("Camera Error", "Cannot open webcam.")
            self.create_main_widgets()
            return

        frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.cam_width = frame_width // 2
        self.cam_height = frame_height // 2

        canvas_width = self.cam_width + 200
        canvas_height = self.cam_height + 200

        total_height = canvas_height + 150
        total_width = canvas_width + 20
        self.master.geometry(f"{total_width}x{total_height}")

        self.video_canvas = tk.Canvas(self.master, width=canvas_width, height=canvas_height)
        self.video_canvas.pack()

        train_button = tk.Button(self.master, text="Train", command=self.open_capture_window, width=20)
        train_button.pack(pady=10)

        cancel_button = tk.Button(self.master, text="Cancel", command=self.cancel_registration, width=20)
        cancel_button.pack(pady=5)

        self.is_streaming = True
        self.update_video_stream()

    def update_video_stream(self):
        if not self.is_streaming or self.video_capture is None:
            return

        ret, frame = self.video_capture.read()
        if not ret:
            self.stop_video_stream()
            messagebox.showerror("Camera Error", "Failed to capture video frame.")
            return

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, **self.detection_params)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2)

        small_frame = cv2.resize(frame, (self.cam_width, self.cam_height))

        cv2image = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)

        x_coord = (self.cam_width + 200) // 2
        y_coord = ((self.cam_height + 200) // 4) + 100

        self.video_canvas.delete("all")
        self.video_canvas.create_image(x_coord, y_coord, anchor=tk.CENTER, image=imgtk)
        self.video_canvas.imgtk = imgtk

        self.master.after(30, self.update_video_stream)

    def cancel_registration(self):
        self.stop_video_stream()
        self.create_main_widgets()

    def open_capture_window(self):
        self.stop_video_stream()
        self.capture_popup = Toplevel(self.master)
        self.capture_popup.title("Capture and Train Faces")
        self.capture_popup.geometry("350x420")
        self.capture_popup.transient(self.master)
        self.capture_popup.grab_set()

        label = Label(self.capture_popup, text=f"Capturing images for {self.user_name} (ID: {self.p_id})", font=("Helvetica", 12))
        label.pack(pady=10)

        self.capture_img_label = Label(self.capture_popup)
        self.capture_img_label.pack()

        self.progress_label = Label(self.capture_popup, text="Captured 0 / 200 images", font=("Helvetica", 10))
        self.progress_label.pack(pady=5)

        self.stop_capture_button = Button(self.capture_popup, text="Stop Capture", command=self.stop_capture, width=20)
        self.stop_capture_button.pack(pady=10)

        self.capture_video = cv2.VideoCapture(0)
        if not self.capture_video.isOpened():
            messagebox.showerror("Camera Error", "Cannot open webcam.")
            self.capture_popup.destroy()
            self.create_main_widgets()
            return

        self.capture_count = 0
        self.max_capture = 200
        self.is_capturing = True
        self.capture_images_loop()

        self.capture_popup.protocol("WM_DELETE_WINDOW", self.on_capture_popup_close)

    def capture_images_loop(self):
        if not self.is_capturing:
            return

        ret, frame = self.capture_video.read()
        if not ret:
            messagebox.showerror("Camera Error", "Failed to capture video frame.")
            self.stop_capture()
            return

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, **self.detection_params)

        if len(faces) > 0 and self.capture_count < self.max_capture:
            (x, y, w, h) = faces[0]
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            img_filename = f"{self.p_id}.{self.capture_count + 1}.jpg"
            img_path = os.path.join(self.data_dir, img_filename)
            cv2.imwrite(img_path, face_img)

            self.capture_count += 1
            self.progress_label.config(text=f"Captured {self.capture_count} / {self.max_capture} images")

            face_color = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGBA)
            img = Image.fromarray(face_color)
            imgtk = ImageTk.PhotoImage(image=img)
            self.capture_img_label.imgtk = imgtk
            self.capture_img_label.configure(image=imgtk)

            if self.capture_count >= self.max_capture:
                self.is_capturing = False
                self.capture_video.release()
                self.capture_video = None
                self.train_model()
                return
        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        frame_img = Image.fromarray(frame_rgb)
        imgtk_full = ImageTk.PhotoImage(image=frame_img)

        self.capture_img_label.after(30, self.capture_images_loop)
        self.capture_img_label.master.update()

    def train_model(self):
        images, labels = [], []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".jpg"):
                try:
                    img = cv2.imread(os.path.join(self.data_dir, filename), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    p_id_str = filename.split(".")[0]
                    if not p_id_str.isdigit():
                        continue
                    label = int(p_id_str)
                    images.append(img)
                    labels.append(label)
                except Exception:
                    continue

        if len(images) == 0 or len(labels) == 0:
            messagebox.showerror("Training Error", "No images found for training.")
            self.capture_popup.destroy()
            self.create_main_widgets()
            return

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(images, np.array(labels))
        recognizer.write("recognition_model.xml")

        self.update_people_csv()

        messagebox.showinfo("Training Complete", f"User '\{self.user_name}\' registered and model trained successfully.")
        self.capture_popup.destroy()
        self.create_main_widgets()

    def update_people_csv(self):
        csv_path = "people.csv"
        file_exists = os.path.exists(csv_path)
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(["p_id", "name"])
                writer.writerow([self.p_id, self.user_name])
        except Exception as e:
            messagebox.showerror("File Error", f"Failed to update people.csv: {e}")

    def stop_capture(self):
        self.is_capturing = False
        if self.capture_video:
            self.capture_video.release()
            self.capture_video = None
        messagebox.showinfo("Capture Stopped", "Image capture stopped.")
        if self.capture_popup:
            self.capture_popup.destroy()
        self.create_main_widgets()

    def on_capture_popup_close(self):
        if messagebox.askokcancel("Quit", "Capture in progress. Do you want to stop?"):
            self.stop_capture()

def main():
    root = tk.Tk()
    root.eval('tk::PlaceWindow . center')
    app = Application(root)
    root.mainloop()

if __name__ == "__main__":
    main()
