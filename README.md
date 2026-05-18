# AI-based Classroom Attendance using Face Recognition

### NAME:ISWARYA P
### REG NO:212223230082

## AIM
To develop an AI-based classroom attendance system using face recognition.

## FEATURES
- Automatic attendance
- Face detection
- Attendance storage
- Live webcam support

## TECHNOLOGIES USED
- Python
- OpenCV
- Face Recognition
- NumPy

## STEPS

### STEP 1: Dataset Collection

Collect student face images using a webcam or uploaded photos and store them with student names/IDs.

### STEP 2: Face Detection

Use OpenCV to detect faces from classroom images or live video frames.

### STEP 3: Face Recognition

Train the system using face encodings with the face_recognition library to identify students.

### STEP 4: Attendance Marking

Compare detected faces with stored student data and automatically mark attendance as Present.

### STEP 5: Database Storage

Store attendance records, student details, date, and time in an SQLite/MySQL database.

### STEP 6: Dashboard Development

Create a web dashboard using Flask where teachers can:

View attendance
Edit/correct records
Export reports as CSV

### STEP 7: Report Generation

Generate attendance reports and export them for academic use.


## PROGRAM
```
import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime

path = 'students'
images = []
classNames = []

myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []

        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')

            f.writelines(f'\n{name},{dtString},Present')

encodeListKnown = findEncodings(images)
print("Encoding Complete")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgs)
    encodesCurFrame = face_recognition.face_encodings(imgs, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):

        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6,y2-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

            markAttendance(name)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
## SAMPLE OUTPUT
Attendance marked automatically using AI.

## RESULT
The AI-based classroom attendance system successfully detected and recognized student faces from live webcam input. Attendance was automatically marked as Present with date and time stored in the database and CSV file.
