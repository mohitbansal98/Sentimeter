import io
import os
import cv2
import re
import imutils

likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
						'LIKELY', 'VERY_LIKELY')

likelihood_sym1 = ('NA', '|', '||', '|||',
					   '||||', '|||||')

likelihood_sym = ('NA', '--|', '-|', '|',
					   '|+', '|++')
score = (0,-2,-1,0,1,2)
face_cascade = cv2.CascadeClassifier("Essential-files/haarcascade_frontalface_alt.xml")
credential_path = "C:/Users/mywor/Desktop/Dubhacks 2020/auth.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
cap = cv2.VideoCapture("Videos/Trim_intrigue.mp4")
# Imports the Google Cloud client library
from google.cloud import vision

# Instantiates a client
client = vision.ImageAnnotatorClient()
font = cv2.FONT_HERSHEY_SIMPLEX
# The name of the image file to annotate
out = cv2.VideoWriter('outpy4_trim.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 28, (1000,531))

while True:
	ret, frame = cap.read()

	if ret == False: # means frame not captured properly
		continue

	frame = imutils.resize(frame, width=1000)
	print(frame.shape)

	cv2.imwrite("frame.jpg", frame) 
	# file_name = os.path.abspath('organized/afraid/{image_name}.jpg'.format(image_name = name))

	# Loads the image into memory
	with io.open("frame.jpg", 'rb') as image_file:
		content = image_file.read()
	image = vision.Image(content=content)

	# response = client.label_detection(image=image)
	response = client.face_detection(image=image)
	faces = response.face_annotations


	angerScore = 0
	joyScore = 0
	sorrowScore = 0
	surpriseScore = 0

	GAP = 15
	for face in faces:
		anger = ('anger: {}'.format(likelihood_sym[face.anger_likelihood]))
		angerScore += score[face.anger_likelihood]
		joy = ('joy: {}'.format(likelihood_sym[face.joy_likelihood]))
		joyScore += score[face.joy_likelihood]
		surprise = ('surprise: {}'.format(likelihood_sym[face.surprise_likelihood]))
		surpriseScore += score[face.surprise_likelihood]
		sorrow = ('sorrow: {}'.format(likelihood_sym[face.sorrow_likelihood]))
		sorrowScore += score[face.sorrow_likelihood]
		vertices = ([(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices])
		text = anger + "\n" + joy + "\n" + surprise + "\n" + sorrow
		x,y =vertices[0]
		# cv2.putText(frame,anger,(x,y),font,0.5,(255, 0, 0),2,0) 
		# cv2.putText(frame,joy,(x,y+GAP),font,0.5,(255, 0, 0),2,0) 
		# cv2.putText(frame,surprise,(x,y+(2*GAP)),font,0.5,(255, 0, 0),2,0) 
		# cv2.putText(frame,sorrow,(x,y+(3*GAP)),font,0.5,(255, 0, 0),2,0) 

	
	length = 150
	height = 20
	coordJoy = (0,0)
	coordSurp = (0,42)
	xLenJoy = int((joyScore * length * 1.0)/ (len(faces)*2))
	xLenSurp = int((surpriseScore * length * 1.0)/ (len(faces)*2))
	color = (0,0,255)

	cv2.rectangle(frame, coordJoy, (coordJoy[0]+length, coordJoy[1]+height), color, 1)
	cv2.rectangle(frame, coordJoy, (coordJoy[0]+xLenJoy, coordJoy[1]+height), color, -1)
	cv2.putText(frame,"happy", (length+5,15), font, 0.5, color,2,0)	

	cv2.rectangle(frame, coordSurp, (coordSurp[0]+length, coordSurp[1]+ height), color, 1)
	cv2.rectangle(frame, coordSurp, (coordSurp[0]+xLenSurp, coordSurp[1]+height), color, -1)	
	cv2.putText(frame,"intrigued", (length+5,57), font, 0.5, color,2,0)

	# print(text)
	# print(vertices)
	out.write(frame)
	cv2.imshow("video frame", frame)

	key_pressed = cv2.waitKey(1) & 0xFF # 0xFF is 11111111, so this operation 
	if key_pressed == ord('q'): # ord('q') gives ascii value
		break

cap.release()
out.release()
cv2.destroyAllWindows()