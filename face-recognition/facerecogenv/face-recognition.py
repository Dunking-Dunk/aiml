import face_recognition
import cv2
import numpy as np
from deepface import DeepFace
import math
from retinaface import RetinaFace
from PIL import Image
import matplotlib.pyplot as plt

models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
]

backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"]

metrics = ["cosine", "euclidean", "euclidean_l2"]

img_path = "./data/group.jpg"
img = cv2.imread(img_path)

resp = RetinaFace.detect_faces(img_path)

for key in resp.keys():
    identity = resp[key]

    facial_area = identity["facial_area"]

    cv2.rectangle(
        img,
        (facial_area[2], facial_area[3]),
        (facial_area[0], facial_area[1]),
        (255, 255, 255),
        1,
    )

cv2.imshow("img", img)
cv2.waitKey(0)


DeepFace.stream("./data", model_name=models[0])


# result = DeepFace.verify(
#     "./data/group.jpg",
#     "./data/dinesh.jpg",
#     detector_backend=backends[4],
#     model_name=models[0],
#     distance_metric=metrics[2],
# )

# print(result)


# DeepFace.stream(db_path="C:/User/Sefik/Desktop/database")

# video_capture = cv2.VideoCapture(0)


# hursun_image = face_recognition.load_image_file("./data/hursun.jpg")
# hursun_face_encoding = face_recognition.face_encodings(hursun_image)[0]


# known_face_encodings = [hursun_face_encoding]
# known_face_names = ["hursun"]


# face_locations = []
# face_encodings = []
# face_names = []
# process_this_frame = True

# while True:
#     ret, frame = video_capture.read()

#     if process_this_frame:
#         small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

#         # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#         rgb_small_frame = small_frame[:, :, ::  -1]

#         # Find all the faces and face encodings in the current frame of video
#         face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
#         face_encodings = face_recognition.face_encodings(
#             rgb_small_frame, face_locations
#         )

#         face_names = []
#         for face_encoding in face_encodings:
#             # See if the face is a match for the known face(s)
#             matches = face_recognition.compare_faces(
#                 known_face_encodings, face_encoding
#             )
#             name = "Unknown"

#             # If a match was found in known_face_encodings, just use the first one.
#             # if True in matches:
#             #     first_match_index = matches.index(True)
#             #     name = known_face_names[first_match_index]

#             # Or instead, use the known face with the smallest distance to the new face
#             face_distances = face_recognition.face_distance(
#                 known_face_encodings, face_encoding
#             )
#             best_match_index = np.argmin(face_distances)
#             if matches[best_match_index]:
#                 name = known_face_names[best_match_index]

#             face_names.append(name)

#     process_this_frame = not process_this_frame

#     for (top, right, bottom, left), name in zip(face_locations, face_names):
#         top *= 4
#         right *= 4
#         bottom *= 4
#         left *= 4

#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#         cv2.rectangle(
#             frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
#         )
#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

#     cv2.imshow("Video", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break


# video_capture.release()
# cv2.destroyAllWindows()
