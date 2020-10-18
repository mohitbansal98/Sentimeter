import io
import os
import cv2
import re
import imutils

likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
						'LIKELY', 'VERY_LIKELY')

likelihood_sym = ('NA', '|', '||', '|||',
					   '||||', '|||||')

likelihood_sym1 = ('NA', '--|', '-|', '|',
					   '|+', '|++')
face_cascade = cv2.CascadeClassifier("Essential-files/haarcascade_frontalface_alt.xml")
credential_path = "C:/Users/mywor/Desktop/Dubhacks 2020/auth.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
cap = cv2.VideoCapture("Videos/zoom_1.mp4")
# Imports the Google Cloud client library
from google.cloud import vision

# Instantiates a client
client = vision.ImageAnnotatorClient()
font = cv2.FONT_HERSHEY_SIMPLEX
# The name of the image file to annotate

while True:
	ret, frame = cap.read()

	if ret == False: # means frame not captured properly
		continue

	frame = imutils.resize(frame, width=1000)

	cv2.imwrite("frame.jpg", frame) 
	# file_name = os.path.abspath('organized/afraid/{image_name}.jpg'.format(image_name = name))

	# Loads the image into memory
	with io.open("frame.jpg", 'rb') as image_file:
		content = image_file.read()
	image = vision.Image(content=content)

	# response = client.label_detection(image=image)
	response = client.face_detection(image=image)
	faces = response.face_annotations


	GAP = 15
	for face in faces:
		anger = ('anger: {}'.format(likelihood_sym[face.anger_likelihood]))
		joy = ('joy: {}'.format(likelihood_sym[face.joy_likelihood]))
		surprise = ('surprise: {}'.format(likelihood_sym[face.surprise_likelihood]))
		sorrow = ('sorrow: {}'.format(likelihood_sym[face.sorrow_likelihood]))
		vertices = ([(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices])
		text = anger + "\n" + joy + "\n" + surprise + "\n" + sorrow
		x,y =vertices[0]
		x-=50
		y+=100
		cv2.putText(frame,anger,(x,y),font,0.5,(0, 0, 0),2,0) 
		cv2.putText(frame,joy,(x,y+GAP),font,0.5,(0, 0, 0),2,0) 
		cv2.putText(frame,surprise,(x,y+(2*GAP)),font,0.5,(0, 0, 0),2,0) 
		cv2.putText(frame,sorrow,(x,y+(3*GAP)),font,0.5,(0, 0, 0),2,0) 

	
	# print(text)
	# print(vertices)
	
	cv2.imshow("video frame", frame)

	key_pressed = cv2.waitKey(1) & 0xFF # 0xFF is 11111111, so this operation 
	if key_pressed == ord('q'): # ord('q') gives ascii value
		break

cap.release()
cv2.destroyAllWindows()