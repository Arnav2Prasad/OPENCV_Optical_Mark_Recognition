# import image using imread() and display it in window

import cv2
import numpy as np
import help_me


# define all needed parameters
widthImg = 700
heightImg = 700

path = "1.jpg"
img = cv2.imread(path)

questions=5
choices = 5
ans = [1,2,0,1,4]
webcamFeed = True
camera_no = 0

cap = cv2.VideoCapture(camera_no)
cap.set(10,150)

while True:
    if webcamFeed : success , img  = cap.read() 
    else: img = cv2.imread(path)

    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    #                   PREPROCESSING



    # the default image os quite big so resize

    img = cv2.resize(img , (widthImg , heightImg))

    imgContours = img.copy()
    imgBiggestContours = img.copy()
    imgFinal = img.copy()

    # now wiull convert to gray scale add blur and then detect edges , and get the rectangle where all markings
    # is present

    # gray
    imgGray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

    # add blur and define size of kernel
    imgBlur = cv2.GaussianBlur(imgGray ,(5,5),1)

    # now detect edges using func
    # define threshold values
    # Canny edge detector
    imgCanny = cv2.Canny(imgBlur , 10,50)

    # use stacking fun to create array of images
    # imageArray =   ([img , imgGray , imgBlur , imgCanny])

    # scale 0.5 and we dont want to chenge the labels for now
    # imgStacked = help_me.stackImages(imageArray , 0.5)


    # in edge detector
    # one box for grading
    # we are able to see diff between marked answer and unmarked and thats a good sign


    try:
        # --------------------------------------------------------
        #           FINDING ALL COUNTOURS
        # RETR_EXTERNAL : will help to find external edges
        # CHAIN_APPROX_NONE : we dont need any approximation
        # now we have to find contours

        countours , hierarchy = cv2.findContours(imgCanny , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)

        # now we have to display the contours
        # define index = -1 : we need all of them
        # define color : 0,255,0 --> green
        # thickness : 10
        cv2.drawContours(imgContours , countours , -1 ,(0,255,0) ,10)

        # we want blank images
        imgBlank = np.zeros_like(img)

        # use stacking fun to create array of images
        # imageArray =   ([img , imgGray , imgBlur , imgCanny],[imgContours,imgBlank,imgBlank,imgBlank])


        # # scale 0.5 and we dont want to chenge the labels for now
        # imgStacked = help_me.stackImages(imageArray , 0.5)


        #               FIND RECTANGLES

        # we want to identify that has the most markings in it and the grade box
        # find which of these are actually rectangle
        # so create a new function 

        rectCon = help_me.rectCountour(countours)

        # biggestContour = rectCon[0]
        # print(biggestContour)

        # but we are getting lots of points
        # can check by len(biggestContour)

        # now try to find conrner points
        biggestContour = help_me.getCornerPoints(rectCon[0])
        gradePoints = help_me.getCornerPoints(rectCon[1])
        # print('----------------------------------------')
        print(biggestContour)

        # now check if they have een detected or not
        if (biggestContour.size != 0 and gradePoints.size != 0):
            # now we can draw them s=to see what we are getting
            # thickness = 10
            # 0 255 0 green
            cv2.drawContours(imgBiggestContours , biggestContour , -1 , (0,255,0),20)

            # 255 0 0 : blue
            cv2.drawContours(imgBiggestContours , gradePoints , -1 , (255,0,0),20)

            # npw we have to get the point
            # so we have to reorder the points
            biggestCountour = help_me.reorder(biggestContour)
            gradePoints = help_me.reorder(gradePoints)

            # apply 
            # define point first and then get the matrix
            pt1 = np.float32(biggestCountour)
            pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])

            matrix = cv2.getPerspectiveTransform(pt1,pt2)

            imgWarpColor = cv2.warpPerspective(img , matrix , (widthImg,heightImg))


            # now repeat the same for grade box
            ptG1 = np.float32(gradePoints)
            ptG2 = np.float32([[0,0],[325,0],[0,150],[325,150]])

            matrixG = cv2.getPerspectiveTransform(ptG1,ptG2)

            imgGradeDisplay = cv2.warpPerspective(img , matrixG , (325,150))


            # now we have to find the markings(answers)

            #               APPLY THRESHOLD
            # the bubbles that dont have markings will have less amount of pixels (applying threshold)
            imgWarpGray = cv2.cvtColor(imgWarpColor , cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpGray , 170 , 255  , cv2.THRESH_BINARY_INV)[1]

            # at markings we will have lot of pixels
            # now will traverse each bubbles by splitting the images into 25 peices
            # images are matrices so for splittig use numpy fun
            boxes = help_me.splitBoxes(imgThresh)

            # index starts from 0!
            # cv2.imshow("Test" , boxes[2])

            # so we check individually whether they are marked or not
            # count non zero pixels

            # eg : you can see the diff
            # print(cv2.countNonZero(boxes[1]))  // output is 10269  ---> so marked
            # print(cv2.countNonZero(boxes[2]))  // output is 2688  --> so unmrked

            #           GETTING NON ZERO PIXEL VALUE OF EACH BOX
            # so traverse and get the max number of pixels element
            # we need array of predefined size ie here 5*5

            myPixelVal = np.zeros((questions,choices))
            count_c = 0
            count_r = 0
            for image in boxes:
                totalPixels = cv2.countNonZero(image)
                myPixelVal[count_r][count_c] = totalPixels

                count_c += 1
                if (count_c == choices):
                    count_c = 0
                    count_r += 1

                
            # matrix storing pixel count for each option          
            print(myPixelVal)

            #           FINDING INDEXX VALUES OF THE MARKINGS
            my_index = []
            for x in range(0,questions):
                arr = myPixelVal[x]
                myIndexVal = np.where(arr == np.amax(arr))
                
                # we got the marked answers
                # print(myIndexVal[0])

                my_index.append(myIndexVal[0][0])
            # marked answers ; 
            # print(my_index)
            
            # now we have to compare this with the original answers

            #           GRADING
            grading=[]
            for x in range(0,questions):
                if ans[x] == my_index[x]:
                    grading.append(1)
                else:
                    grading.append(0)


            # print gradings
            # print(grading)

            #           FINAL SCORE
            score = (sum(grading)/questions) * 100  # final grade
            print(score) 

            # now we have to display the score , put green if correct , red if wrong

            #           DISPLAYING ANSWERS
            imgResult =imgWarpColor.copy()
            imgResult = help_me.showAnswers(imgResult , my_index , grading , ans , questions , choices)

            # now we have to put the colors back to the original answer sheet
            imgRawDrwaing = np.zeros_like(imgWarpColor )
            imgRawDrwaing = help_me.showAnswers(imgRawDrwaing , my_index , grading , ans , questions , choices)

            # now we take this image do inverse and put in original image



            # will create Inverse matrix 
            inv_matrix = cv2.getPerspectiveTransform(pt2,pt1)

            imgInWarp = cv2.warpPerspective(imgRawDrwaing , inv_matrix , (widthImg,heightImg))



            imgRawGrade = np.zeros_like(imgGradeDisplay)
            # scale : 3
            # thickness : 3
            cv2.putText(imgRawGrade , str(int(score)) + "%" , (50,100) , cv2.FONT_HERSHEY_COMPLEX , 3 , (0,255,255),3)
            # cv2.imshow("Grade",imgRawGrade )
            
            inv_matrixG = cv2.getPerspectiveTransform(ptG2,ptG1)

            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade , inv_matrixG , (widthImg,heightImg))
            # now combine to get the final image
            # combine usin added weights
            imgFinal = cv2.addWeighted(imgFinal,1,imgInWarp,1,0)
            imgFinal = cv2.addWeighted(imgFinal,1,imgInvGradeDisplay,1,0)





        imgBlank = np.zeros_like(img)
        # use stacking fun to create array of images
        imageArray =   ([img , imgGray , imgBlur , imgCanny],[imgContours,imgBiggestContours,imgWarpColor,imgThresh]
                        ,[imgResult,imgRawDrwaing,imgInWarp,imgFinal])
    except:
        imgBlank = np.zeros_like(img)
        # use stacking fun to create array of images
        imageArray =   ([img , imgGray , imgBlur , imgCanny],[imgBlank,imgBlank,imgBlank,imgBlank]
                        ,[imgBlank,imgBlank,imgBlank,imgBlank])



    # LABELS FOR DISPLAY
    lables = [["Original","Gray","Blur","Canny"],["Contours","Biggest Con","Warpped","imgThres"],
            ["Result" , "Raw Drawing" , "Inv Warp" , "Final"]]

    # scale 0.5 and we dont want to chenge the labels for now
    # imgStacked = help_me.stackImages(imageArray , 0.3 , lables)
    imgStacked = help_me.stackImages(imageArray , 0.4)


    # cv2.drawContours
    cv2.imshow("Final Result" , imgFinal)
    cv2.imshow("Stacked Images" , imgStacked)
    # name the window original
    # cv2.imshow("Original",img)
    # cv2.waitKey(0)    

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("FinalResult.jpg" , imgFinal)
        cv2.waitKey(300)
