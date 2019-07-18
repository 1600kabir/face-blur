import cv2

def blur(filename):
	image = cv2.imread(filename)  
	result_image = image.copy()
	face_cascade_name = "./haarcascade_frontalface_default.xml"

	face_cascade = cv2.CascadeClassifier()
	face_cascade.load(face_cascade_name)

	grayimg = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

	faces = face_cascade.detectMultiScale(grayimg, 1.1, 2, 0|cv2.CASCADE_SCALE_IMAGE, (30, 30))

	if len(faces) != 0: 
		for f in faces:
			x, y, w, h = [ v for v in f ]

			sub_face = image[y:y+h, x:x+w]

			sub_face = cv2.GaussianBlur(sub_face,(23, 23), 30)

			result_image[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face
		
		cv2.imwrite('./result.jpg', result_image)

blur('img.jpg')
