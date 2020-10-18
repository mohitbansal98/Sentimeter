# import os

# init_path = "Data/KDEF_and_AKDEF/KDEF"

# emotion = {
# "AF" : "afraid", 
# "AN" : "angry", 
# "DI" : "disgusted", 
# "HA" : "happy", 
# "NE" : "neutral", 
# "SA" : "sad",
# "SU" : "surprised"
# }

# for filename in os.listdir(init_path):
# 	new_path = init_path + "/" + filename
# 	print(filename)
# 	for file in os.listdir(new_path):
# 		if file[-5:-4] == "S":
# 			print("    " + file + "-->" + emotion[file[4:6]])

		for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            print "face found in file: %s" %f
            gray = gray[y:y+h, x:x+w] #Cut the frame to size
            try:
                out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
                cv2.imwrite("dataset\\%s\\%s.jpg" %(emotion, filenumber), out) #Write image
            except:
               pass #If error, pass file