import cv2
import numpy as np
import os
import datetime

# os.chdir("C:\\Users\\zjx61\\Desktop")
# img=cv2.imread("waitToProcess.png")

def bek(img,i,j,k):
    img[i][j][0]=k # B>220
    img[i][j][1]=k # G>220
    img[i][j][2]=k # R<20

def line(img):
    img1=img.copy()

    for i in range (img1.shape[0]):
        for j in range (img1.shape[1]):
            if(img[i][j][0]>254 and img[i][j][1]>254 and img[i][j][2]<1):
                bek(img1,i,j,255)
            else:
                bek(img1,i,j,0)
    # compose=np.hstack((img,img1))
    # cv2.imshow("compose",compose)
    '''膨胀
    kernel = np.ones((2,2),np.uint8)
    dst = cv2.dilate(img1,kernel)
    compose=np.hstack((img1,dst))
    cv2.imshow("compose",compose)
    '''

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    draw_img = img1.copy()
    # res = cv2.drawContours(draw_img, contours[3], -1, (0, 0, 255), 2)
    img_res=np.empty(img.shape)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        [vx, vy, x4, y4] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        x1,x2,y1,y2=x,x+w,y,y+h
        a,b=x2-x1,y2-y1
        c,d=x2-x1,y1-y2
        xmin,ymin,xmax,ymax=0,0,0,0
        if abs(a*vy-b*vx)<=abs(c*vy-d*vx):
            xmin,ymin,xmax,ymax=x,y,x+w,y+h
        else:
            xmin,ymin,xmax,ymax=x,y+h,x+w,y
        a,b=xmax-xmin,ymax-ymin
        # if vx*a+vy*b<0:
        #     xmin,ymin,xmax,ymax=xmax,ymax,xmin,ymin
        img_res = cv2.line(img_res,(xmin, ymin),(xmax, ymax),(0,255,0),2)
    img_line=img.copy()
    for i in range (img_res.shape[0]):
        for j in range (img_res.shape[1]):
            if(img_res[i][j][0]!=0.0 or img_res[i][j][1]!=0.0 or img_res[i][j][2]!=0.0):
                img_line[i][j]=img_res[i][j]
    return img_line

os.chdir("C:\\大三下\\大创\\计算机图像处理\\lane")
cap = cv2.VideoCapture('videofile_maksed_20210515T191543.avi')
size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
codec = cv2.VideoWriter_fourcc(*'DIVX')
save_dir = os.path.join(os.getcwd(), "output")
file_name = "videofile_maksed_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
file_name = os.path.join(save_dir, file_name)
print(save_dir)
print(file_name)
output = cv2.VideoWriter(file_name, codec, 60.0, size)

while(cap.isOpened()):
    ret, img = cap.read()
    img_line=line(img)
    cv2.imshow("res.jpg",img_line)
    output.write(img_line)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()

