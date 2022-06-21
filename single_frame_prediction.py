from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# load json and create model
json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model/model.h5")
print("Loaded model from disk")



# checking prediction
classes = ['bad','best','glad','sad','scared','stiff','surprise']

img = image.load_img("images/best.jpg",target_size=(224,224))
img_ = np.asarray(img)
print(img_.shape)
plt.figure(figsize=(8,5))
plt.imshow(img)

img = np.expand_dims(img_, axis=0)
output = loaded_model.predict(img)
x=np.argmax(output, axis = 1)
print(classes[int(x)])
plt.title(f"predicted image : {classes[int(x)]}")
plt.show()