import cv2
vid = cv2.VideoCapture(0)
body_classifier = cv2.CascadeClassifier('haardcascade_fullbody.xml')


while(True):
   
    
    ret, frame = vid.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(grey,1.2,3)
    for (x,y,w,h) in bodies: 
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        save = frame[y:y+h,x:x+w]
        cv2.imwrite("Isaac.png",save)
    
    cv2.imshow("Web cam", frame)
      
   
    if cv2.waitKey(25) == 32:
        break
  

vid.release()


cv2.destroyAllWindows()