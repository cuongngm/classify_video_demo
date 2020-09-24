from tensorflow.keras.models import load_model
from collections import deque
import cv2
import pickle
import numpy as np


model = load_model("sport/sport_model.h5")
lb = pickle.loads(open("sport/lb.pickle", "rb").read())

mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=128)

i = 0
label = "Predicting..."

vs = cv2.VideoCapture("sport/f1_train_1.mp4")
frame_width = int(vs.get(3))
frame_height = int(vs.get(4))

fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
out = cv2.VideoWriter('sport/output2.avi', fourcc, 10, (frame_width, frame_height))

while True:
    ret, frame = vs.read()
    if not ret:
        break

    i += 1
    display = frame.copy()
    if i % 10 == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224)).astype("float32")
        frame -= mean
        # predict
        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)
        # average
        result = np.array(Q).mean(axis=0)
        i = np.argmax(result)
        label = lb.classes_[i]

    text = "Watching: {}".format(label)
    cv2.putText(display, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("output", display)
    out.write(display)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break
vs.release()
out.release()
cv2.destroyAllWindows()
