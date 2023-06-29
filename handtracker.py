# First let import all the modules we need
import time
from threading import Thread
import cv2 as cv
import mediapipe as mp
import os
from PIL import Image

# Define some variables
x1, y1 = 0, 0  # This will hold the position for the index finger of the first hand
x2, y2 = 0, 0  # This will hold the position for the index finger of the second hand
shown = False  # If true , then the
take_screenshot = False  # If true then take screenshot
allowed = False  # if true then a screenshot is allowed to be taken
minimum_area_to_take_screenshot = 10000  # This is the minimum pixel area to be exceeded in order for it to allow a screenshot to be taken
seconds_to_wait_and_take_screenshot = 5  # I think 5 is too big
threshold = 5

"""
This function contains an infinite loop 
It waits for the number of seconds contained in the seconds_to_wait_and_take_screenshot variable, 
which is every 5 seconds then it sets the take_screenshot variable to true.
This function is called in another thread so that the infinite while loop will not block the main thread from running.
"""


def take_screenshot_func():
    global take_screenshot
    while True:
        take_screenshot = True
        time.sleep(seconds_to_wait_and_take_screenshot)


# Initialize a new thread to call our """loop function""".
screenshot_timer = Thread(target=take_screenshot_func)
screenshot_timer.start()  # Start the thread

"""
Define a folder to store all the screenshots.
If the folder does not exists in our working directory then create it
"""
folder_to_store_screenshots = "ScreenShots"
folder_to_store_merged_images = "Merged Images"

if not os.path.exists(folder_to_store_screenshots):
    os.mkdir(folder_to_store_screenshots)
if not os.path.exists(folder_to_store_merged_images):
    os.mkdir(folder_to_store_merged_images)

# This text file will help us remember the current screenshot number . E.g screenshot3.png, screenshot4.png
# The idea is to update the file every time in our loop .
if not os.path.exists(".remember.txt"):
    with open(".remember.txt", "w") as first:
        first.write("0")
if not os.path.exists(".remember2.txt"):
    with open(".remember2.txt", "w") as first:
        first.write("1")

# Read the last screenshot number stored in the file and assign it to a variable n.
with open(".remember.txt") as remembered:
    n = int(str(remembered.read().strip()))
with open(".remember2.txt") as remembered:
    n2 = int(str(remembered.read().strip()))
"""
Define a class HandTracker which will help us reuse code efficiently by defining some methods and attributes
"""


class HandTracker:
    def __init__(self, max_hands=2, static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.max_hands = max_hands
        self.mode = static_image_mode
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands  # Create a mediapipe hands solutions object. There is also a face solutions class and as well a pose solutions class. So you get it.
        self.mpDraw = mp.solutions.drawing_utils  # Also create a mediapipe drawing object for drawing landmark connections and so on.

        """
        Define a Hand object which takes in some arguments 
        like the maximum number of hands to detect, static_image_mode
        should be set to False since we are reading from a video stream . 
        It tells mediapipe to keep track of the hands from frame to frame 
        and not just as a single picture which has no relationship with another picture.
        """
        self.hands = self.mpHands.Hands(max_num_hands=self.max_hands,
                                        static_image_mode=self.mode,
                                        min_detection_confidence=self.min_detection_confidence,
                                        min_tracking_confidence=self.min_tracking_confidence, )
        self.draw = False  # This will determine whether to draw landmark connections and so on.
        self.frame = None  # This is the actual image frame
        self.frameRGB = None
        self.first_hand = []
        self.second_hand = []
        """Mediapipe accepts only RGB images 
        and since opencv uses BGR color space , we need 
        to convert the BGR image to RGB so this variable holds the RGB image  """

    """This method takes in the image frame and returns 
    the new image frame, the coordinates of the first hand 
    index finger, the coordinates of the second hand index finger"""

    def findIndexFingersForBothHands(self, image, draw=True):
        self.frame = image
        self.frameRGB = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)  # Convert image from BGR to RGB

        """Get the results which is a mediapipe solutions object 
        and have an attribute called multi_hand_landmarks 
        which gives information about the hands detected"""
        self.results = self.hands.process(self.frameRGB)
        self.draw = draw

        # Lists to store the x and y coordinates of the index finger of each hand
        first_index_finger = []
        second_index_finger = []
        self.first_hand = []
        self.second_hand = []

        if self.results.multi_hand_landmarks:  # Check to see if there is any results
            for _id, handLms in enumerate(
                    self.results.multi_hand_landmarks):  # loop through the points collecting its index and landmark. Check out the mediapipe documentation to see the image of the hand and the particular index for a point
                for point, lm in enumerate(handLms.landmark):
                    h, w, c = self.frame.shape
                    cx, cy = int(lm.x * w), int(
                        lm.y * h)  # lm.x and lm.y returns the x,y position of the in a format where 0.5 for instance represents half of the screen width or height.So we need to convert then to actual pixel coordinates according to the dimentions of our image (height and width)

                    # Point 8 is the index finger for any hand so we get its x and y coordinates
                    if point == 8:
                        if _id == 0:  # If its the first hand
                            first_index_finger = [cx, cy]
                        elif _id == 1:  # If its the second hand
                            second_index_finger = [cx, cy]

                    if _id == 0:
                        self.first_hand.append([point, cx, cy])
                    elif _id == 1:
                        self.second_hand.append([point, cx, cy])

                # If the draw attribute is true ,  we want to draw the landmarks and the connections as well.
                if self.draw:
                    self.mpDraw.draw_landmarks(
                        self.frame, handLms, self.mpHands.HAND_CONNECTIONS,
                        connection_drawing_spec=self.mpDraw.DrawingSpec(color=self.mpDraw.RED_COLOR, thickness=4),
                        landmark_drawing_spec=self.mpDraw.DrawingSpec(color=self.mpDraw.RED_COLOR, thickness=2)
                    )
        # Now we return the new frame, first hand's index finger's x and y position and second hand's index finger's x and y position
        return self.frame, first_index_finger, second_index_finger

    def allFingersFromEitherHandsUp(self):
        lms = []
        if self.first_hand and not self.second_hand:
            lms = self.first_hand
        elif self.second_hand and not self.first_hand:
            lms = self.second_hand
        fingers = []
        finger_points = [4, 8, 12, 16, 20]
        pt = []
        real_pt = []

        if lms:
            for finger_point in finger_points:
                finger_for_point = lms[finger_point]
                if finger_for_point[2] <= lms[finger_point - 1][2]:
                    pt.append(1)
            real_pt = pt
            pt = []
            return real_pt


"""
THis function is called passing in the pdfpath and the image path,
which then now adds the image to the pdf and if the append variable is true the image is added on a separate page.
"""


def addImageToPdfFile(pdfPath, imagePath):
    img = Image.open(imagePath)  # load the image
    # Convert image chanel to rbg format
    img = img.convert("RGB")  # Convert the image to RGB channel
    if os.path.exists(pdfPath):
        append = True
    else:
        append = False
    img.save(pdfPath, append=append, subject="Merged Images")  # Save the image to the pdf file


if __name__ == "__main__":
    cam = cv.VideoCapture(0)  # Create a videocapture object
    tracker = HandTracker()  # Create an object or instance of the HandTracker class to track the hands
    fingers_up = []

    while True:
        _, frame = cam.read()
        frame = cv.flip(frame, 1)
        # Unpack all the values returned by the findIndexFingersForBothHands method setting draw to true

        # Get the fingers that are up for any hand
        fingers_up = tracker.allFingersFromEitherHandsUp()

        frame, index1, index2 = tracker.findIndexFingersForBothHands(frame, draw=True)
        h, w, s = frame.shape  # get the height, width and color channel of the image

        """
        Now the idea here is as the function loops in another 
        thread(I'm talking about the infinite while loop running in another thread), 
        every 5 seconds it sets the take_screenshot variable to true. 
        Then if the index finger from the first hand shows 
        up(so if the index1 variable is not empty), x1 and y1 are assigned its x,y position.
        Then it the index finger from the second hand shows up(also if the index2 
        variable is not empty), x2 and y2 are assigned its x,y position.
        
        Then the x1 , x2 and y1,y2 values are normalized so that an error is not thrown.
        Then a rectangle is drawn from x1,y1 to x2,y2
        Also the screenshot is calculated using frame[initialRow:finalRow, initialColumn:finalColumn]. 
        Note that the semi-colon(:) means from initialRow TO finalRow.
        So, after getting the screenshot A.K.A REGION OF INTEREST (ROI), 
        the shown variable is set to true which means we have gotten our roi. 
        Then the roi's area is calculated and if it exceeds the threshold value (which is set 
        in the minimum_area_to_take_screenshot variable) , allowed is set to true (which means we are 
        allowed to take a screenshot.)
        Next we check if the take_screenshot , shown and allowed variables are all true, 
        then we write our frameRoi to an image file (save it), increment the image count, update our remember file which stores the last count
        show the screenshot image in a separate window and set all those variables to false. 
        Note that all of those were wrapped inside a try and except block because if the selected screenshot are is too small,  an exception is thrown.
        Then finally we handle the exception and display the final frame
        """
        if index1:  # Check if at least one hand is up
            if fingers_up and len(fingers_up) == 5:  # check if all the five fingers are up
                last_one = n

                images = os.listdir(
                    f"{folder_to_store_screenshots}/")  # Get all the images currently in the screenshots folder and store them into a list.

                """
                Now the number of images in the folder must reach a certain number in other to merge them.
                This is defined in the threshold variable.
                """
                if len(images) >= threshold:
                    first_nine = images[
                                 :threshold]  # If there are enough images to merge then we get the first number of images defined in the threshold variable
                    image_mat = [cv.imread(f"{folder_to_store_screenshots}/{image}") for image in
                                 first_nine]  # We now use opencv to read the images and store their corresponding arrays in the list
                    merged = cv.vconcat(
                        image_mat)  # We now merge all the image arrays vertically into one image array assigned to the merged variable
                    cv.imwrite(f"{folder_to_store_merged_images}/merged_{n2}.png",
                               merged)  # We save the merged image in the merged images folder
                    [os.remove(f"{folder_to_store_screenshots}/{image_path}") for image_path in
                     first_nine]  # Finally we use a list comprehension to loop through all the images on the merged one then delete them as well
                    cv.imshow("Merged", merged)  # We now show the merged image
                    n = 0
                    image_path = f"{folder_to_store_merged_images}/merged_{n2}.png"  # we now update the image path
                    n2 += 1
                    addImageToPdfFile("images.pdf",
                                      image_path)  # Then we call the function to add the merged image to the pdf file , and the rest of the program remains the same as before.
                else:
                    print("Not enough values waiting to complete...")

            else:
                x1, y1 = index1

        if index2:
            x2, y2 = index2
            if x1 < x2:
                x2, x1 = x1, x2
            if y1 < y2:
                y2, y1 = y1, y2
            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv.LINE_8)
            frameROI = frame[y2:y1, x2:x1]
            shown = True
            frameRoiArea = int(abs(x2 - x1) * abs(y2 - y1))
            cv.putText(frame, f"Screenshot Area: {frameRoiArea}", (int(0.1 * w), int(0.9 * h)),
                       cv.FONT_HERSHEY_SIMPLEX, 1,
                       (255, 255, 0), 2)  # Display the area on the screen

            if frameRoiArea >= minimum_area_to_take_screenshot:
                allowed = True

        if take_screenshot and shown and allowed:
            try:
                frameROI = cv.resize(frameROI, (480, 360))
                cv.imwrite(f"{folder_to_store_screenshots}/screenshot_{n + 1}.png", frameROI)

                n += 1
                with open(".remember.txt", "w") as to_remember:
                    to_remember.write(str(n))

                cv.imshow("ROI", frameROI)
                shown = False
                take_screenshot = False
                allowed = False

            except cv.error:
                cv.putText(frame, "Image is too small", (int(0.1 * w), int(0.7 * h)), cv.FONT_HERSHEY_SIMPLEX, 1,
                           (255, 0, 0), 2)

        cv.imshow("Image", frame)
        if cv.waitKey(1) & 0xff == ord("d"):
            break
    cam.release()
    cv.destroyAllWindows()
