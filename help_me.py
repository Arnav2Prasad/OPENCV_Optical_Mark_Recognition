import cv2
import numpy as np

## TO STACK ALL THE IMAGES IN ONE WINDOW
# we have created many intermediate images to get the desired output , 
# now for review purpose we want to see all imgs
# so murtaza has created a function for it(not explained in yt vid , used previously)
def stackImages(imgArray,scale,lables=[]):

    # input : give array of images , define scale and labels
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        #print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

def rectCountour(countours):
    # we dont want any small rectangles
    # chck if it has 4 corner points and use it find the area and get the largest

    rectCon = []
    for i in countours:
        area = cv2.contourArea(i)
        # print(area)

        # some of the area are very small so we want to detect it 
        if (area>50):
            # True --> for closed objects
            peri = cv2.arcLength(i,True)

            # now check how many corner points it has
            # resolution = 0.02*peri
            # true : closed 
            approx = cv2.approxPolyDP(i,0.02*peri , True)
            # to check how many points it has : len(approx)
            # print("Corner Points ",approx)

            if (len(approx)==4):
                rectCon.append(i)
    # list of all rect contours
    # print(rectCon)

    # want to arrange them based on their area
    # sorting based on  key = cv2.contourArea
    rectCon = sorted(rectCon , key=cv2.contourArea,reverse=True)

    return rectCon
                

# define corners
def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True) # LENGTH OF CONTOUR
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True) # APPROXIMATE THE POLY TO GET CORNER POINTS
    return approx

def reorder(myPoints):

    myPoints = myPoints.reshape((4, 2)) # REMOVE EXTRA BRACKET
    # print(myPoints)
    myPointsNew = np.zeros((4, 1, 2), np.int32) # NEW MATRIX WITH ARRANGED POINTS

    # decide at which axis to sum the points
    add = myPoints.sum(1)

    # print(add)
    # print(np.argmax(add))
    myPointsNew[0] = myPoints[np.argmin(add)]  #[0,0]
    myPointsNew[3] =myPoints[np.argmax(add)]   #[w,h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]  #[w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)] #[h,0]

    return myPointsNew


def splitBoxes(img):
    # define how many cuts to make ie 5 here 
    # verticalsplit   
    rows = np.vsplit(img,5)
    boxes=[]
    for r in rows:
        # pre define column split ie 5 here
        # horizontal split
        cols= np.hsplit(r,5)
        for box in cols:
            boxes.append(box)
    return boxes

# img : image that we want to display on
def showAnswers(img,myIndex,grading,ans,questions=5,choices=5):
     
    # to get width and height of each choice to mark on that point
     secW = int(img.shape[1]/questions)
     secH = int(img.shape[0]/choices)

     for x in range(0,questions):
         myAns= myIndex[x]

        #  to know center position
         cX = (myAns * secW) + secW // 2
         cY = (x * secH) + secH // 2

         if grading[x]==1:
            myColor = (0,255,0)
            #cv2.rectangle(img,(myAns*secW,x*secH),((myAns*secW)+secW,(x*secH)+secH),myColor,cv2.FILLED)
            cv2.circle(img,(cX,cY),50,myColor,cv2.FILLED)
         else:
            myColor = (0,0,255)
            #cv2.rectangle(img, (myAns * secW, x * secH), ((myAns * secW) + secW, (x * secH) + secH), myColor, cv2.FILLED)
            cv2.circle(img, (cX, cY), 50, myColor, cv2.FILLED)

            # CORRECT ANSWER
            myColor = (0, 255, 0)
            correctAns = ans[x]
            cv2.circle(img,((correctAns * secW)+secW//2, (x * secH)+secH//2),
            20,myColor,cv2.FILLED)
     return img
