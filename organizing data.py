import cv2
import os

face_decector_1 = cv2.CascadeClassifier("Essential-files/haarcascade_frontalface_default.xml")
face_decector_2 = cv2.CascadeClassifier("Essential-files/haarcascade_frontalface_alt.xml")
face_decector_3 = cv2.CascadeClassifier("Essential-files/haarcascade_frontalface_alt2.xml")
face_decector_4 = cv2.CascadeClassifier("Essential-files/haarcascade_frontalface_alt_tree.xml")

emotion = {
"AF" : "afraid", 
"AN" : "angry", 
"DI" : "disgusted", 
"HA" : "happy", 
"NE" : "neutral", 
"SA" : "sad",
"SU" : "surprised"
}

init_path = "Data/KDEF_and_AKDEF/KDEF"
filenumber = 0
for filename in os.listdir(init_path):
	new_path = init_path + "/" + filename
	for file in os.listdir(new_path):
		if file[-5:-4] == "S":
			filePath = new_path + "/" + file
			print(filePath)
			curr_emotion = emotion[file[4:6]]
			frame = cv2.imread(filePath)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			face = face_decector_1.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
			face_two = face_decector_2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
			face_three = face_decector_3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
			face_four = face_decector_4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

			facefeatures = ""
			if len(face) == 1:
				facefeatures = face
			elif len(face_two) == 1:
				facefeatures = face_two
			elif len(face_three) == 1:
				facefeatures = face_three
			elif len(face_four) == 1:
				facefeatures = face_four
			else:
				facefeatures = ""

			for (x, y, w, h) in facefeatures:
				print("face found in file: %s" %file)
				gray = gray[y:y+h, x:x+w] #Cut the frame to size
				try:
					out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
					# filename = str(filenumber) + ".jpg"
					# path = "organized/" + curr_emotion
					# cv2.imwrite(os.path.join(path , filename), out)
					cv2.imwrite("organized/{emotion}/{name}.jpg".format(emotion = curr_emotion, 
								name = filenumber), out)
					# to see images being saved
					# cv2.imshow('image', out)
					# cv2.waitKey(0)
					# cv2.destroyAllWindows()
				except:
					pass #If error, pass file
			filenumber+=1



