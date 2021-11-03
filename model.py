from PIL import Image
import io
import imutils
import dlib
import cv2
import numpy as np
import xgboost as xg
import lightgbm as lgb
import pickle
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
def rect_to_bb(rect):

	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords

def setInfo(shape):
    features = []
    features.append(np.linalg.norm(shape[17] - shape[21])/np.linalg.norm(shape[0] - shape[16]))
    features.append(np.linalg.norm(shape[22] - shape[26])/np.linalg.norm(shape[0] - shape[16]))
    features.append(np.linalg.norm(shape[36] - shape[39])/np.linalg.norm(shape[0] - shape[16]))
    features.append(np.linalg.norm(shape[42] - shape[45])/np.linalg.norm(shape[0] - shape[16]))
    features.append(np.linalg.norm(shape[31] - shape[35])/np.linalg.norm(shape[0] - shape[16]))
    features.append(np.linalg.norm(shape[27] - shape[33])/np.linalg.norm(shape[0] - shape[16]))
    features.append(np.linalg.norm(shape[48] - shape[54])/np.linalg.norm(shape[0] - shape[16]))
    features.append(np.linalg.norm(shape[60] - shape[64])/np.linalg.norm(shape[0] - shape[16]))
    features.append(np.linalg.norm(shape[27] - shape[8])/np.linalg.norm(shape[0] - shape[16]))

    return features


def worker(img_bytes):
    image = Image.open(io.BytesIO(img_bytes))
    open_cv_image = np.array(image)  
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    image = imutils.resize(open_cv_image, width=224)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("data")

    rects = detector(gray, 1)

    if(len(rects)==0 or len(rects)>1):
        return []
    
    
    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        features = setInfo(shape)

    return features

def predict(img_bytes):
    features = worker(img_bytes)
    if len(features)==0:
        return []
    features = np.array(features)
    features = np.expand_dims(features,axis=0)


    model1 = lgb.Booster(model_file='./models/best_lgbm.txt')
    print("MODEL 1 LOADED")
   
    print("HERE??????????")
    model2 = xg.XGBRegressor()
    model2.load_model("./models/best_xgboost.txt")
    print("MODEL 2 LOADED")
    model3 = pickle.load(open("./models/best_nn.sav", 'rb'))
    print("MODEL 3 LOADED")
    meta_model = pickle.load(open("./models/best_model_SVM.sav", 'rb'))


    y_pred_model1 = model1.predict(features)
    y_pred_model2 = model2.predict(features)
    y_pred_model3 = model3.predict(features)

    features = np.vstack((y_pred_model1,y_pred_model2,y_pred_model3))
    features = features.transpose()


    y_pred = meta_model.predict(features)
    
    return y_pred
