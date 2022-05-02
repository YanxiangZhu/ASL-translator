import tkinter as tk
from tkinter import messagebox
import cv2
from tkinter import *
from PIL import Image,ImageTk
import numpy as np
import mediapipe as mp
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from keras.models import load_model

def main():
    wind=tk.Tk()
    wind.geometry("800x400")
    wind.title("Sign Language Detect System")

    canvas = tk.Canvas(wind, height=400, width=800)
    login_background = Image.open("bc1.png").resize((800, 400))
    login_background = ImageTk.PhotoImage(login_background)
    login_image = canvas.create_image(0, 0, anchor='nw', image=login_background)
    canvas.pack(side='top')

    title_lab=tk.Label(wind,text="Welcome To Our Sign Language Detect System",bg="yellow",font="仿宋 17 bold")
    title_lab.place(x=140,y=30)

    userlab=tk.Label(wind, text="username", font="仿宋 20 bold", fg="blue", width=8)
    userlab.place(x=200,y=100)
    user_entry=tk.Entry(wind, width=15,bg="white",font="仿宋 20 bold")
    user_entry.place(x=350,y=100)

    sslab=tk.Label(wind, text="password", font="仿宋 20 bold", fg="blue", width=8)
    sslab.place(x=200,y=200)
    ss_entry=tk.Entry(wind, width=15,bg="white",font="仿宋 20 bold",show="*")
    ss_entry.place(x=350,y=200)

    def login():
        username = user_entry.get()
        password = ss_entry.get()
        if username == 'admin' and password == 'admin':
            print('login success.')
            messagebox.showinfo("info", "login success.")
            wind.destroy()
            gui = Gui()
        else:
            print('username or password error, please check it!')
            messagebox.showinfo("info", "username or password error, please check it!")

    login_btn=tk.Button(wind,text="Login", font="仿宋 20 bold", fg="blue", width=8, command=login)
    login_btn.place(x=350,y=300)
    wind.mainloop()


class Gui(object):
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.camera = 0
        self.map = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
            4: "E",
            5: "F",
            6: "G",
            7: "H",
            8: "I",
            9: "J",
            10: "K",
            11: "L",
            12: "M",
            13: "N",
            14: "O",
            15: "P",
            16: "Q",
            17: "R",
            18: "del",
            19: "S",
            20: "T",
            21: "U",
            22: "V",
            23: "W",
            24: "X",
            25: "Y",
            26: "Z",
            27: "nothing",
            28: "space"
        }
        self.model = self.load_model()
        self.root = Tk()
        self.root.title("Sign Language Detect System")
        self.canvas = Canvas(self.root, bg='#c4c2c2', width=600, height=400)
        self.canvas.pack(padx=10, pady=10)
        self.root.config(cursor="arrow")
        self.btn = Button(self.root, text="open the camera", command=self.openCamera1)
        self.btn.pack(fill="both", expand=True, padx=10, pady=10)
        self.btn1 = Button(self.root, text="close the camera", command=self.exist)
        self.btn1.pack(fill="both", expand=True, padx=11, pady=10)
        self.root.mainloop()

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False  # Image is no longer writeable
        results = model.process(image)  # Make prediction
        image.flags.writeable = True  # Image is now writeable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
        return image, results

    def draw_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION)  # Draw face connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                                  self.mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                                  self.mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections

    def draw_styled_landmarks(self, image, results):
        # Draw face connections
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                                  self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )
        # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                  self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )
        # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                  self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )
        # Draw right hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                  self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

    def load_model(self):
        model = load_model("model_f.h5")
        return model

    def openCamera(self):
        self.camera = cv2.VideoCapture(0)
        while True:
            success, img = self.camera.read()
            img = cv2.flip(img, 1)
            (h, w) = img.shape[:2]
            print(h)
            print(w)
            if success:
                cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
                current_image = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=current_image)
                self.canvas.create_image(0, 0, anchor='nw', image=imgtk)
            self.root.update_idletasks()

            self.root.update()

    def getMapper(self, label):
        return self.map.get(label)

    def openCamera1(self):
        self.camera = cv2.VideoCapture(0)
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.camera.isOpened():
                ret, frame = self.camera.read()
                if frame is None:
                    break
                (h, w) = frame.shape[:2]
                width = 1200
                r = width / float(w)
                dim = (600, 400)
                frame1 = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                # Make detections
                image, results = self.mediapipe_detection(frame1, holistic)
                # print(results)

                # Draw landmarks
                self.draw_styled_landmarks(image, results)

                dim2 = (200, 200)
                frame2 = cv2.resize(frame, dim2, interpolation=cv2.INTER_AREA)
                frame2 = np.array(frame2).reshape(1, 200, 200, 3)
                # gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                # print(gray2.shape)
                prediction = self.model.predict(frame2)
                results = [np.argmax(prediction[i]) for i in range(len(prediction))]
                predict_label = self.getMapper(results[0])

                cv2.putText(image, "Predict action: {}".format(predict_label), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                key = cv2.waitKey(10) & 0xFF

                if key == 27:
                    break
                # cv2image = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGBA)
                current_image = Image.fromarray(image)
                imgtk = ImageTk.PhotoImage(image=current_image)
                self.canvas.create_image(0, 0, anchor='nw', image=imgtk)
                self.root.update_idletasks()
                self.root.update()

        self.root.mainloop()
        # self.camera.release()
        # cv2.destroyAllWindows()

    def exist(self):
        self.camera.release()
        cv2.destroyAllWindows()
        self.root.quit()

    def __del__(self):
        self.camera.release()
        cv2.destroyAllWindows()
        self.root.quit()

if __name__ == '__main__':
    main()
