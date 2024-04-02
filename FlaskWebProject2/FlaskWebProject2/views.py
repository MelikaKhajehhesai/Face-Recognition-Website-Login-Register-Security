"""
Routes and views for the flask application.
"""

from cgitb import reset
from datetime import datetime
from email.charset import BASE64
from logging import exception
import re
from flask import render_template
from FlaskWebProject2 import app

from flask import request, jsonify

import json

import base64

import PatternGenerator

@app.route('/signin', methods=['GET','POST'])
def signin():
    if request.method == 'POST':

        print(request.form['file'])
        b = request.form['file']
        b = b.replace("data:image/jpeg;base64,", "")
        b = base64.b64decode(b)
        f = open('face.jpg' ,'wb')
        f.write(b)
        f.close()
        result , username = FaceRecognizing('face.jpg')
        if( result):
            return render_template('result.html', message = f"{username}, Your have been Loogedin Successfully...!")
        else:
            return render_template('result.html', message = "Error: Sorry, You can not access this website.")

def Face_Detect(img):
    classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.3,5)

    if len(faces) == 0:
        return img, []

    for (x,y,w,h) in faces:
        cv2.rectangle(img ,(x,y),(x+w, y+h),(0,255,0),2)
        croppedimage = img[y:y+h , x:x+w]
        croppedimage = cv2.resize(croppedimage, (200,200))
    return img, croppedimage, (x,y)


def FaceRecognizing(imagefilename):
    frame = cv2.imread(imagefilename)
    image, frame , (x,y) = Face_Detect(frame)
    face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    model = cv2.face.LBPHFaceRecognizer_create()

    dirs =  os.listdir('dataset')
    print(dirs)
    for dir in dirs:
        model.read(f'dataset/{dir}/user_face_pattern.xml')
        ######### TAAA INJAAAAA

        result = model.predict(face)
    
        if result[1]<500:
             confidence = int(100 * (1- (result[1]/300)))
             if confidence > 70:
                return True, dir
        #     else:
        #        return False
        #else:
        #    return False
    return False, 'nouser'


@app.route("/LoginWithImage", methods=['post'])
def LoginWithImage():
    d = request.data
    print(d)
    #my_json = d.decode("utf8").replace("'", '"')
    #print(my_json)
    print(d)
    #j = json.loads(my_json)
    j = json.loads(d)
    print(j["imageData"])

    #files=listdir('dataset')
    #print(files)

    #models=[]
    
    
    #content = j["imageData"].replace("data:image/png;base64,", "")
    b = base64.b64decode(j["imageData"])  # j['video'])
    fs = open("login.png", "wb")   
    fs.write(b)
    fs.close()



import cv2
import numpy as np
from os import listdir 


@app.route("/UploadVideoFileForLogin", methods=["POST"])
def UploadVideoFileForLogin():
    d = request.data
    my_json = d.decode("utf8").replace("'", '"')
    print(my_json)
    j = json.loads(my_json)
    print(j["video"])
    username = j['username']
    print(username)

    content = j["video"].replace("data:video/webm;base64,", "")
    b = base64.b64decode(content)  # j['video'])

    try:
        os.removedirs('dataset/' +username)
    except:
        pass
    try:
        os.mkdir('dataset/' +username)
    except:
        pass
    fs = open(f"video.webm", "wb")
    fs.write(b)
    fs.close()
    Generate_DatabaseFiles(f"video.webm", username)

    GenerateHaarCasecadePatternFile(username)#"video.webm")
    print(b)
    return


@app.route("/UploadVideoFile", methods=["POST"])
def UploadVideoFile():
    d = request.data
    my_json = d.decode("utf8").replace("'", '"')
    print(my_json)
    j = json.loads(my_json)
    username = j['username']
    print(username)
    print(j["video"])

    content = j["video"].replace("data:video/webm;base64,", "")
    b = base64.b64decode(content)  # j['video'])
    fs = open("video.webm", "wb")
    fs.write(b)
    fs.close()
    
    return


@app.route('/login')
def Login():
    return render_template('login.html')


@app.route('/')
@app.route("/register")
def Webcam_Storing():
    return render_template("register.html")


@app.route("/webcam")
def Open_Webcam():
    return render_template("webcam.html")


# @app.route('/')
@app.route("/home")
def home():
    """Renders the home page."""
    return render_template(
        "index.html",
        title="Home Page",
        year=datetime.now().year,
    )


@app.route("/contact")
def contact():
    """Renders the contact page."""
    return render_template(
        "contact.html",
        title="Contact",
        year=datetime.now().year,
        message="Your contact page.",
    )


@app.route("/about")
def about():
    """Renders the about page."""
    return render_template(
        "about.html",
        title="About",
        year=datetime.now().year,
        message="Your application description page.",
    )


import cv2
import numpy as np

import cv2
import os

model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def Facecroping(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        face = model.detectMultiScale(gray, 1.1, 20)
    except exception as ex:
        pass

    if face in () or face == []:
        return []
    croping = []
    for x, y, w, h in face:
        croping = img[y : y + h, x : x + w]
    return croping


#def Generate_DatabaseFiles(filename):
#    camera = cv2.VideoCapture(filename)
#    count = 1
#    while True:
#        status, img = camera.read()
#        if status == False: break
#        if status == True:
#            faces = Facecroping(img)
#            if faces == []:
#                continue
#            faces = cv2.resize(faces, (200, 200))
#            faces = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)

#            cv2.imwrite(f"dataset/user_{count}.jpg", faces)
#            #faces = cv2.putText(
#            #    faces, str(count), (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2
#            #)
#            #cv2.imshow("crop face", img)  # faces)
#            count += 1
#        if cv2.waitKey(1) == 13 or count > 300:
#            break
#    #camera.release()
#    #cv2.destroyAllWindows()
#    str= "END"

import os
def GenerateHaarCasecadePatternFile(username):
    files = os.listdir('dataset/' +f'{username}')
    print(files)
    tarningdata = []
    label = []
    for i,filename in enumerate(files):
       image=cv2.imread( 'dataset/' +f'{username}/' + filename , cv2.IMREAD_GRAYSCALE) 
       tarningdata.append(np.asarray(image,dtype=np.uint8)) 
       label.append(i)
    tarningdata=np.asarray(tarningdata)
    label=np.asarray(label,dtype= np.int32) 
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(tarningdata,label)
    print(f'Training has been finished seccesfully')
    
    #i = 0
    #while True:
    #    status, frame = camera.read()
    #    if status == True:
    #        # print(frame)
    #        tarningdata.append(np.asarray(frame, dtype=np.uint8))
    #        label.append(i)
    #        i += 1
    #    else:
    #        break
    #tarningdata = np.asarray(tarningdata)
    #label = np.asarray(label, dtype=np.int32)
    #model = cv2.face.LBPHFaceRecognizer_create()
    #model.train(tarningdata, label)
    model.save(f"dataset/{username}/user_face_pattern" + ".xml")
    print(f"pattern seccesfully")


def Generate_DatabaseFiles(filename, username):
    camera = cv2.VideoCapture(f'{filename}')
    count = 1
    while True:
        status, img = camera.read()
        if status == False: break
        if status == True:
            faces = Facecroping(img)
            if len(faces) == 0:
                continue
            faces = cv2.resize(faces, (200, 200))
            faces = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)

            cv2.imwrite('dataset/' +f"{username}/user_{count}.jpg", faces)
            #faces = cv2.putText(
            #    faces, str(count), (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2
            #)
            #cv2.imshow("crop face", img)  # faces)
            count += 1
        if cv2.waitKey(1) == 13 or count > 300:
            break
    #camera.release()
    #cv2.destroyAllWindows()
    str= "END"