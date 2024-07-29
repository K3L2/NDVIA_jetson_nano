import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import Label, Button, ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
messagebox.showinfo("device info", f"{device} 로 작동중 입니다.")

# 전역 변수 초기화
model = None
model_names = None
capture_image = None
processed_image = None
save_dir = "./save_img"
model_dir = "./src/yolo_model"
cap = None

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def color_space_converted_save(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def single_scale_retinex(image, sigma=30):
    def ssr(img, sigma):
        retinex = np.log10(img + 1) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1)
        return retinex

    image = image / 255.0
    retinex_image = ssr(image, sigma)
    retinex_image = (retinex_image - np.min(retinex_image)) / (np.max(retinex_image) - np.min(retinex_image))
    retinex_image = (retinex_image * 255).astype(np.uint8)
    return retinex_image


def apply_preprocessing(image, method):
    if method == 'normal':
        return image
    elif method == 'hsv':
        return color_space_converted_save(image)
    elif method == 'retinex':
        return single_scale_retinex(image)
    else:
        return image


def yolo_obj_draw(img, threshold=0.3):
    detection = model(img)[0]
    boxinfos = detection.boxes.data.tolist()

    for data in boxinfos:
        x1, y1, x2, y2 = map(int, data[:4])
        confidence_score = round(data[4], 2)
        classid = int(data[5])
        name = model_names[classid]
        if confidence_score > threshold:
            model_text = f'{name}_{confidence_score}'
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, text=model_text, org=(x1, y1 - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.9, color=(255, 0, 0), thickness=2)


def update_frame():
    global capture_image, preprocessed_image
    ret, frame = cap.read()
    if ret:
        capture_image = frame.copy()
        if model is not None:
            processed_image = capture_image.copy()
            preprocessed_image = apply_preprocessing(processed_image, preprocessing_method.get())
            yolo_obj_draw(preprocessed_image, threshold=0.2)
            image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image)
        webcam_label.imgtk = image_tk
        webcam_label.configure(image=image_tk)
    webcam_label.after(10, update_frame)


def load_model():
    global model, model_names
    selected_model = model_combobox.get()
    if selected_model and selected_model != "normal":
        model_path = os.path.join(model_dir, selected_model)
        model = YOLO(model_path)
        model.to(device)  # Ensure the model uses GPU
        model_names = model.names
        messagebox.showinfo("Model Loaded", f"{selected_model} 모델을 불러왔습니다")
    else:
        model = None
        model_names = None
        messagebox.showinfo("Model Unloaded", "YOLO 모델이 실행되지 않습니다.")


def save_image():
    global capture_image, preprocessed_image
    count = len(os.listdir(save_dir)) + 1
    if model is not None and preprocessed_image is not None:
        filename = os.path.join(save_dir, f"capture_{count:03d}.jpg")
        cv2.imwrite(filename, preprocessed_image)
        print(f"Image saved: {filename}")
    elif capture_image is not None:
        filename = os.path.join(save_dir, f"webcam_{count:03d}.jpg")
        cv2.imwrite(filename, capture_image)
        print(f"Image saved: {filename}")
    else:
        messagebox.showwarning("No Image Captured", "캡처된 이미지가 없습니다")


def key_event(event):
    if event.keysym == 'Return':
        save_image()


def start_webcam():
    global cap
    selected_cam = cam_combobox.get()
    if selected_cam:
        cap_index = int(selected_cam.split(" ")[1])
        cap = cv2.VideoCapture(cap_index, cv2.CAP_DSHOW)
        update_frame()


def stop_webcam():
    global cap
    if cap is not None:
        cap.release()
        cap = None
        webcam_label.config(image='')

    control_frame.pack_forget()
    init_control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)


def find_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.read()[0]:
            break
        else:
            arr.append(f"Camera {index}")
        cap.release()
        index += 1
    return arr


# 모델 파일 리스트 가져오기
model_files = ["normal"] + [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]
available_cameras = find_cameras()

# GUI 생성
root = tk.Tk()
root.title("YOLO Webcam GUI")
root.bind('<Return>', key_event)

webcam_label = Label(root)
webcam_label.pack(side=tk.LEFT)

capture_label = Label(root)
capture_label.pack(side=tk.LEFT)

init_control_frame = tk.Frame(root)
init_control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

Label(init_control_frame, text="사용할 웹캠을 선택한 후 start를 누르세요").grid(row=0, column=0, padx=5, pady=5)
cam_combobox = ttk.Combobox(init_control_frame, values=available_cameras)
cam_combobox.grid(row=1, column=0, padx=5, pady=5)

start_button = Button(init_control_frame, text="Start", command=lambda: [init_control_frame.pack_forget(),
                                                                         control_frame.pack(side=tk.BOTTOM, fill=tk.X,
                                                                                            padx=10, pady=10),
                                                                         start_webcam()])
start_button.grid(row=2, column=0, padx=5, pady=5)

control_frame = tk.Frame(root)

model_combobox = ttk.Combobox(control_frame, values=model_files)
model_combobox.grid(row=0, column=0, padx=5, pady=5)

load_button = Button(control_frame, text="Load Model", command=load_model)
load_button.grid(row=1, column=0, padx=5, pady=5)

# 전처리 방법 선택
Label(control_frame, text="전처리 방법을 선택하세요").grid(row=2, column=0, padx=5, pady=5)
preprocessing_method = ttk.Combobox(control_frame, values=['normal', 'hsv', 'retinex'])
preprocessing_method.grid(row=3, column=0, padx=5, pady=5)
preprocessing_method.current(0)

Label(control_frame, text="Enter 키를 누르면 사진이 저장됩니다").grid(row=4, column=0, padx=5, pady=5)

stop_button = Button(control_frame, text="Stop", command=stop_webcam)
stop_button.grid(row=5, column=0, padx=5, pady=5)

root.mainloop()

# Release the webcam and close windows when the GUI is closed
if cap is not None:
    cap.release()
cv2.destroyAllWindows()
