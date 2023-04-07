import cv2
import os

# path to classifier file
cascPath = "haarcascade_frontalface_alt.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
pics = 0 # if saving images, comment this if that is not necessary

path = os.getcwd()
dataPath = path + "\Data\\" # Location of input images, mine was in subfolder Data. Change to your desired path

resultPath = path + "\Results\\" # Path to save images to. Change to desired path
for f in os.listdir('Data'):
    faceCoords = []
    pics += 1 # if saving images, comment this if that is not necessary
    count = 0 # if saving images, comment this if that is not necessary

    # Read the image
    image = cv2.imread(os.path.join(dataPath, f))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image given the properties
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # Get a cropped version of the faces
    for (x, y, w, h) in faces:
        crop_img = image[y:y+h, x:x+w] # cropped image for emtion detection

        # top left and bottom right of face
        faceCoords.append([x,y,x+w,y+h]) # coords of faces for crowd detection

        # Comment next part if pipelining!
        # START SAVING IMAGE PART!!!!!-------------------------
        
        name = str(pics) + "_" + str(count) + ".jpg"
        isExist = os.path.exists(resultPath)
        if not isExist:
            print("making new dir")
            # Create a new directory because it does not exist
            os.makedirs(resultPath)

        # Save image
        cv2.imwrite(os.path.join(resultPath, name), crop_img)
        count += 1

        # END SAVING IMAGE PART!!!!!----------------------------