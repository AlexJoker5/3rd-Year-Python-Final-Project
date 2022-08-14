from tkinter import *
from tkinter import filedialog
import os
import cv2
import pandas as pd
from PIL import ImageTk,Image
from utils import image_resize

window = Tk()
window.title("Final Project")
window.iconbitmap('UsedImages/favicon.ico')
window.geometry("960x640")
window.minsize(960,640)
window.maxsize(1366,768)

def file_upload():
    answer = filedialog.askopenfile(parent=f1,
                                    initialdir=os.getcwd(),
                                    title="Please select Image/Video:")
    Label(f1, text=answer.name, foreground='white', font=('Courier', 10), bg="grey").place(x=110, y=145)
    return answer.name

def color_detection():
    img_path = file_upload()
    csv_path = 'colors.csv'

    index = ['color', 'color_name', 'hex', 'R', 'G', 'B']
    df = pd.read_csv(csv_path, names=index, header=None)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (800, 600))

    r = g = b = xpos = ypos = 0

    def get_color_name(R, G, B):
        minimum = 1000
        for i in range(len(df)):
            d = abs(R - int(df.loc[i, 'R'])) + abs(G - int(df.loc[i, 'G'])) + abs(B - int(df.loc[i, 'B']))
            if d <= minimum:
                minimum = d
                cname = df.loc[i, 'color_name']

        return cname

    clicked = False

    while True:
        def draw_function(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                global b, g, r, xpos, ypos, clicked
                clicked = True
                xpos = x
                ypos = y
                b, g, r = img[y, x]
                b = int(b)
                g = int(g)
                r = int(r)
                cv2.rectangle(img, (20, 20), (600, 60), (b, g, r), -1)
                text = get_color_name(r, g, b) + ' R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)
                cv2.putText(img, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                if r + g + b >= 600:
                    cv2.putText(img, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_function)
        cv2.imshow('image', img)

        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

def objectDetection():
    w2 = Tk()
    w2.title("Object Detection")
    w2.iconbitmap("UsedImages/favicon.ico")
    w2.geometry("650x500")
    w2.resizable(0, 0)
    Button(w2, text="Object Detection with masks", command=lambda:[w2.destroy(), anotherODInterface_mask()], font=('Courier', 15)).place(x=175, y=100)
    Button(w2, text="Object Detection with mustaches", command=lambda:[w2.destroy(), anotherODInterface_moustaches()], font=('Courier', 15)).place(x=150, y=200)
    Button(w2, text="Object Detection with glasses", command=lambda:[w2.destroy(), anotherODInterface_glasses()] , font=('Courier', 15)).place(x=160, y=300)
    btnClose = Button(w2, text="Close", command=w2.destroy, font=15).place(x=560,y=440)

def anotherODInterface_mask():
    w3 = Tk()
    w3.title("Object Detection With Face Mask")
    w3.iconbitmap("UsedImages/favicon.ico")
    w3.geometry("650x500")
    w3.resizable(0, 0)
    Button(w3, text="Object Detection on WebCam",
           command=object_detection_using_web_cam_mask,font=('Courier', 15)).place(x=150, y=140)
    Button(w3, text="Object Detection on Video",
           command=object_detection_on_video_mask, font=('Courier', 15)).place(
            x=150, y=210)
    btnBack = Button(w3, text="Back", command=lambda: [w3.destroy(), objectDetection()], font=15).place(x=500,y=440)
    btnClose = Button(w3, text="Close", command=w3.destroy, font=15).place(x=560,y=440)

def anotherODInterface_moustaches():
    w4 = Tk()
    w4.title("Object Detection With Moustaches")
    w4.iconbitmap("UsedImages/favicon.ico")
    w4.geometry("650x500")
    w4.resizable(0, 0)
    Button(w4, text="Object Detection on WebCam",
           command=object_detection_using_web_cam_moustaches, font=('Courier', 15)).place(x=150, y=140)
    Button(w4, text="Object Detection on Video",
           command=object_detection_on_video_moustaches, font=('Courier', 15)).place(
        x=150, y=210)
    btnBack = Button(w4, text="Back", command=lambda: [w4.destroy(), objectDetection()], font=15).place(x=500, y=440)
    btnClose = Button(w4, text="Close", command=w4.destroy, font=15).place(x=560, y=440)

def anotherODInterface_glasses():
    w5 = Tk()
    w5.title("Object Detection With Glasses")
    w5.iconbitmap("UsedImages/favicon.ico")
    w5.geometry("650x500")
    w5.resizable(0, 0)
    Button(w5, text="Object Detection on WebCam",
           command=object_detection_using_web_cam_glasses, font=('Courier', 15)).place(x=150, y=140)
    Button(w5, text="Object Detection on Video",
           command=object_detection_on_video_glasses, font=('Courier', 15)).place(
        x=150, y=210)
    btnBack = Button(w5, text="Back", command=lambda: [w5.destroy(), objectDetection()], font=15).place(x=500, y=440)
    btnClose = Button(w5, text="Close", command=w5.destroy, font=15).place(x=560, y=440)

def object_detection_using_web_cam_mask():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    face_mask = cv2.imread('UsedImages/mask.jpg')
    scaling_factor = 1.2

    if face_cascade.empty():
        raise IOError('Unable to load the face cascade classifier xml file')

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        if ret:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for (x, y, w, h) in face_rects:
                if h > 0 and w > 0:
                    h, w = int(1.0 * h), int(1.2 * w)
                    y += 32
                    x -= 15

                    frame_roi = frame[y:y + h, x:x + w]

                    face_mask_small = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_AREA)
                    gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)

                    ret, mask = cv2.threshold(gray_mask, 244, 255, cv2.THRESH_BINARY_INV)
                    mask_inv = cv2.bitwise_not(mask)

                    masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)

                    masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
                    frame[y:y + h, x:x + w] = cv2.add(masked_face, masked_frame)

            cv2.imshow('img', frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break

    cap.release()

    cv2.destroyAllWindows()

def object_detection_on_video_mask():
    cap = cv2.VideoCapture(file_upload())
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    face_mask = cv2.imread('UsedImages/mask.jpg')

    scaling_factor = 0.5

    if face_cascade.empty():
        raise IOError('Unable to load the face cascade classifier xml file')

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        if ret:

            for (x, y, w, h) in face_rects:
                if h > 0 and w > 0:
                    h, w = int(1.0 * h), int(1.2 * w)
                    y += 20
                    x -= 9

                    frame_roi = frame[y:y + h, x:x + w]

                    face_mask_small = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_AREA)
                    gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)

                    ret, mask = cv2.threshold(gray_mask, 244, 255, cv2.THRESH_BINARY_INV)
                    mask_inv = cv2.bitwise_not(mask)

                    masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)

                    masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
                    frame[y:y + h, x:x + w] = cv2.add(masked_face, masked_frame)

            cv2.imshow('img', frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break

    cap.release()

    cv2.destroyAllWindows()

def object_detection_using_web_cam_moustaches():
    mouth_cascade = cv2.CascadeClassifier('xml/haarcascade_mcs_mouth.xml')

    moustache_mask = cv2.imread('UsedImages/moustaches.jpg')

    if mouth_cascade.empty():
        raise IOError('Unable to load the mouth cascade classifier xml file')

    cap = cv2.VideoCapture(0)
    scaling_factor = 1.3

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)

        mouth_rects = mouth_cascade.detectMultiScale(gray, 1.3, 5)
        if len(mouth_rects) > 0:
            (x, y, w, h) = mouth_rects[0]
            h, w = int(1.5 * h), int(1.8 * w)
            x -= 0.23 * w
            x = int(x)
            y -= 0.55 * h
            y = int(y)
            frame_roi = frame[y:y + h, x:x + w]
            moustache_mask_small = cv2.resize(moustache_mask, (w, h), interpolation=cv2.INTER_AREA)

            gray_mask = cv2.cvtColor(moustache_mask_small, cv2.COLOR_BGR2GRAY)

            ret, mask = cv2.threshold(gray_mask, 50, 255, cv2.THRESH_BINARY_INV)
            mask_inv = cv2.bitwise_not(mask)
            masked_mouth = cv2.bitwise_and(moustache_mask_small, moustache_mask_small, mask=mask)
            masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
            frame[y:y + h, x:x + w] = cv2.add(masked_mouth, masked_frame)

        cv2.imshow('Moustache', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def object_detection_on_video_moustaches():
    cap = cv2.VideoCapture(file_upload())
    mouth_cascade = cv2.CascadeClassifier('xml/haarcascade_mcs_mouth.xml')

    moustache_mask = cv2.imread('UsedImages/moustaches.jpg')

    if mouth_cascade.empty():
        raise IOError('Unable to load the mouth cascade classifier xml file')

    scaling_factor = 0.8

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)

        mouth_rects = mouth_cascade.detectMultiScale(gray, 1.3, 5)
        if len(mouth_rects) > 0:
            (x, y, w, h) = mouth_rects[0]
            h, w = int(1.5 * h), int(1.8 * w)
            x -= 0.23 * w
            x = int(x)
            y -= 0.55 * h
            y = int(y)
            frame_roi = frame[y:y + h, x:x + w]
            moustache_mask_small = cv2.resize(moustache_mask, (w, h), interpolation=cv2.INTER_AREA)

            gray_mask = cv2.cvtColor(moustache_mask_small, cv2.COLOR_BGR2GRAY)

            ret, mask = cv2.threshold(gray_mask, 50, 255, cv2.THRESH_BINARY_INV)
            mask_inv = cv2.bitwise_not(mask)
            masked_mouth = cv2.bitwise_and(moustache_mask_small, moustache_mask_small, mask=mask)
            masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
            frame[y:y + h, x:x + w] = cv2.add(masked_mouth, masked_frame)

        cv2.imshow('Moustache', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def object_detection_using_web_cam_glasses():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eyes_cascade = cv2.CascadeClassifier('xml/frontalEyes35x16.xml')
    glasses = cv2.imread("UsedImages/glasses.png", -1)
    scaling_factor = 1.3

    while (True):
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.5, 5)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + h]
            roi_color = frame[y:y + h, x:x + h]

            eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
            for (ex, ey, ew, eh) in eyes:
                glasses2 = image_resize(glasses.copy(), width=ew)

                gw, gh, gc = glasses2.shape
                for i in range(0, gw):
                    for j in range(0, gh):
                        if glasses2[i, j][3] != 0:
                            roi_color[ey + i, ex + j] = glasses2[i, j]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        cv2.imshow('Glasses', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def object_detection_on_video_glasses():
    cap = cv2.VideoCapture(file_upload())
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eyes_cascade = cv2.CascadeClassifier('xml/frontalEyes35x16.xml')
    glasses = cv2.imread("UsedImages/glasses.png", -1)
    scaling_factor = 0.8

    while (True):
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.5, 5)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + h]
            roi_color = frame[y:y + h, x:x + h]

            eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
            for (ex, ey, ew, eh) in eyes:
                glasses2 = image_resize(glasses.copy(), width=ew)

                gw, gh, gc = glasses2.shape
                for i in range(0, gw):
                    for j in range(0, gh):
                        if glasses2[i, j][3] != 0:
                            roi_color[ey + i, ex + j] = glasses2[i, j]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        cv2.imshow('Glasses', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def help_detection():
    Label(f1, text='You can also use image for object detection but not recommend', foreground='white', bg="grey", font=('Courier', 11)).place(x=100,
                                                                                                             y=290)
    Label(f1, text='Upload only image for color detection', foreground='white', bg="grey", font=('Courier', 11)).place(x=100, y=310)
    Label(f1, text='Press Esc to exit from color and object detections output window', foreground='white', bg="grey", font=('Courier', 11)).place(
        x=100, y=330)


img1 = cv2.imread('UsedImages/markus.jpg')
width = 1366
height = 768
dim = (width, height)
resized = cv2.resize(img1, dim)
cv2image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
img = Image.fromarray(cv2image)
imgtk = ImageTk.PhotoImage(image=img)
Label(window,image=imgtk).place(x=-2,y=0)

f1=Frame(window,width=700, height=355,bg='grey')
f1.pack(expand=True)

Label(f1, text='Liberal Arts Program, MIT', foreground='white', bg="grey", font=('Algerian', 17)).place(x=210,y=15)
Label(f1, text='3rd Year, Python, Computer Science', foreground='white', bg="grey", font=('Algerian', 17)).place(x=155, y=55)
Label(f1, text='Group 1 - Christopher Lone Toe, Htwe Myat Cho', foreground='white', bg="grey", font=('Algerian', 15)).place(x=125, y=95)

Label(f1, text='Path : - ', foreground='white', font=('Courier', 10), bg="grey").place(x=40, y=145)

Button(f1, text="Object Detection           ", command=objectDetection, font=('Courier', 15)).place(x=180, y=180)
Button(f1, text="Color Identification       ", command=color_detection, font=('Courier', 15)).place(x=180, y=235)
Button(f1, text="Help", command=help_detection, font=('Courier', 15)).place(x=20, y=300)

window.mainloop()
cv2.destroyAllWindows()
