import io
import os
import re

likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
						'LIKELY', 'VERY_LIKELY')
likelihood_sym = ('NA', '--|', '-|', '|',
					   '|+', '|++')
credential_path = "C:/Users/mywor/Desktop/Dubhacks 2020/auth.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

# Imports the Google Cloud client library
from google.cloud import vision

# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
file_name = os.path.abspath('Data/KDEF_and_AKDEF/KDEF/AF02/AF02SUS.JPG')

# Loads the image into memory
with io.open(file_name, 'rb') as image_file:
	content = image_file.read()

image = vision.Image(content=content)

# response = client.label_detection(image=image)
response = client.face_detection(image=image)
faces = response.face_annotations

for face in faces:
	anger = ('anger: {}'.format(likelihood_name[face.anger_likelihood]))
	joy = ('joy: {}'.format(likelihood_name[face.joy_likelihood]))
	surprise = ('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))
	sorrow = ('sorrow: {}'.format(likelihood_name[face.sorrow_likelihood]))
	print(anger,"\n" ,joy,"\n" ,surprise,"\n" ,sorrow)
	vertices = ([(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices])
	# bounds = list('face bounds: {}'.format(','.join(vertices)))
	print(vertices[0])
