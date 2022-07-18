import cv2
import os

# from FaceRecognation import facedetector

video=cv2.VideoCapture(0)

facedetect=cv2.CascadeClassifier('haarcscade_frontalface_defult.xml')

# count=0

# nameID =str(input("Enter your name ")).lower()

# path = 'image/'+nameID

# isExist = os.path.exits(path)
    
# if isExist:    
#     print("Name alredy taken")
#     nameID = str(input("enter your name again : "))
# else:
#     os.makedirs(path) 

while True:
    ret,frame= video.read()
    faces= facedetect.detectMultiScale(frame,1.3, 5)
    for x,y,w,h in faces:
    #     count=count+1
    #     os.name='./image/'+nameID+ '/'+str(count)+'.jpg'
    #     print("creating Image--------")
    #     cv2.imwrite(os.name,frame[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow("windowframe",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break 
video.release()
cv2.destroyAllWindows()