import cv2
  
  
img = cv2.imread("flower.jpg")
  
# variables

  

  
cv2.destroyAllWindows()

class draw_line:  
  
  def __init__(self):
      
    self.ix = -1
    self.iy = -1
    self.drawing = False
    self.x=-1
    self.y=-1
    self.img=None
    self.thickness=-1

  def draw_rectangle_with_drag(self,event, x, y, flags, param):
    thickness=self.thickness
      
    if event == cv2.EVENT_LBUTTONDOWN:
        self.drawing = True
        self.ix = x
        self.iy = y            
              
    elif event == cv2.EVENT_MOUSEMOVE:
        if self.drawing == True:
            cv2.line(self.img, (self.ix, self.iy),
                          (x, y),
                          (0, 255, 255),
                          thickness)
      
    elif event == cv2.EVENT_LBUTTONUP:
        self.drawing = False
        cv2.line(self.img, (self.x, self.y),
                      (x, y),
                      (0, 255, 255),
                      thickness)
        self.x=x
        self.y=y
  
  def start(img,self):

    self.img=img 
    cv2.namedWindow(winname = "Region of interest")
    cv2.setMouseCallback("Region of interest", self.draw_rectangle_with_drag)
