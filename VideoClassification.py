import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from collections import deque

import matplotlib.pyplot as plt


# load json and create model
json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model/model.h5")
print("Loaded model from disk")
classes = ['bad','best','glad','sad','scared','stiff','surprise']

mean = np.array([123.68,116.779,103.939][::1], dtype='float32')
Queue = deque(maxlen=128)

path = "demo_videos/best.mp4"
video_capture = cv2.VideoCapture(path)
# video_capture = cv2.VideoCapture(0)
writer = None
(Width,Height) = (None,None)
while (video_capture.isOpened()):
    (taken, frame) = video_capture.read()
    if not taken:
        break
    if Width is None or Height is None:
        (Width,Height) = frame.shape[:2]

    output = frame.copy()
    output = cv2.resize(output, (500 ,500))
    # print(output.shape)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.resize(frame, (224,224)).astype('float32')
    frame -=mean
    preds = loaded_model.predict(np.expand_dims(frame, axis=0))[0]
    Queue.append(preds)
    results = np.array(Queue).mean(axis=0)
    prec = round(np.max(results), 2) * 100
    # print(results)
    i=np.argmax(results)
    # i=np.argmax(preds,axis = 1)
    label = classes[int(i)]

    # text = f"Bad:{round(results[0]*100,2)} " \
    #        f"Best:{round(results[1]*100,2)} " \
    #        f"Glad:{round(results[2]*100,2)} " \
    #        f"Sad:{round(results[3]*100,2)} " \
    #        f"Scared:{round(results[4]*100,2)} " \
    #        f"Stiff:{round(results[5]*100,2)} " \
    #        f"Surprise:{round(results[6]*100,2)}"

    cv2.putText(output, label+":"+str(round(results[int(i)]*100,2))+"%", (30, 45), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0))

    if writer is None:
        fourcee = cv2.VideoWriter_fourcc(*'DIVX')
        writer = cv2.VideoWriter(f'outputVideo/{path.split("/")[-1].split(".")[0]}.avi', fourcee, 5, (500,500), True)
    writer.write(output)
    cv2.imshow("In Progress...",output)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

print("Finalizing...")
writer.release()
video_capture.release()
cv2.destroyAllWindows()