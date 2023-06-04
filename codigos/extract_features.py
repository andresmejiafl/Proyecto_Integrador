from sklearn import preprocessing
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import glob
import cv2
import os

def features_vgg16(path):
    """
    Función para utilizar la cnn VGGNET16 como extractor de caracteristicas,
    esta función entrega como resultado las caracteristicas y etiquetado codificado
    para las particiones de train y validación.
    Tiene como unico parametro de entrada la ruta con las imagenes de train y test 
    previamente separadas.
    """
    print('******************************************************************** ')
    print('Feature extraction using vggnet16')
    print('******************************************************************** ')

    SIZE = 256

    train_images = []
    train_labels = []
    test_images  = []
    test_labels  = [] 

    for directory_path in glob.glob(path + "/train/*"):

        label_tr = directory_path.split("\\")[-1]

        for img_path in glob.glob(os.path.join(directory_path, "*.png")):

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
            img = cv2.resize(img, (SIZE, SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            train_images.append(img)
            train_labels.append(label_tr)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    for directory_path in glob.glob(path + "/test/*"):

        label_te = directory_path.split("\\")[-1]

        for img_path in glob.glob(os.path.join(directory_path, "*.png")):

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (SIZE, SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            test_images.append(img)
            test_labels.append(label_te)

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    print("Labels: " + str(np.unique(train_labels)))

    le = preprocessing.LabelEncoder()
    le.fit(train_labels)
    train_labels_encoded = le.transform(train_labels)
    le.fit(test_labels)
    test_labels_encoded = le.transform(test_labels)

    print("Labels encoded: " + str(np.unique(train_labels_encoded)))

    x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded
    x_train, x_test = x_train / 255, x_test / 255

    VGG_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (SIZE, SIZE, 3))

    for layer in VGG_model.layers:
        layer.trainable = False

    X_train_feature = VGG_model.predict(x_train)
    X_train_features = X_train_feature.reshape(X_train_feature.shape[0], -1)

    X_test_feature = VGG_model.predict(x_test)
    X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

    print('******************************************************************** ')
    print('Feature extraction finished')
    print('******************************************************************** ')

    return X_train_features, y_train, X_test_features, y_test


def features_nasnet(path):
    """
    Función para utilizar la cnn nasnet como extractor de caracteristicas,
    esta función entrega como resultado las caracteristicas y etiquetado codificado
    para las particiones de train y validación.
    Tiene como unico parametro de entrada la ruta con las imagenes de train y test 
    previamente separadas.
    """
    print('******************************************************************** ')
    print('Feature extraction using nasnet')
    print('******************************************************************** ')

    train_labels = []
    test_labels  = [] 

    train_data_dir = path + '/train/'
    valid_data_dir = path + '/test/'

    base_model = NASNetLarge(weights='imagenet', include_top=True)

    base_model.trainable = False

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Cargar los conjuntos de datos de entrenamiento y validación
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(331, 331),
                                                        class_mode='categorical')

    valid_generator = valid_datagen.flow_from_directory(valid_data_dir,
                                                        target_size=(331, 331),
                                                        class_mode='categorical')

    pre_train = base_model.predict(train_generator)
    pre_valid = base_model.predict(valid_generator)

    for directory_path in glob.glob(path + "/train/*"):

        label_tr = directory_path.split("\\")[-1]

        for img_path in glob.glob(os.path.join(directory_path, "*.png")):

            train_labels.append(label_tr)

    for directory_path in glob.glob(path + "/test/*"):

        label_te = directory_path.split("\\")[-1]

        for img_path in glob.glob(os.path.join(directory_path, "*.png")):

            test_labels.append(label_te)

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    le = preprocessing.LabelEncoder()
    le.fit(train_labels)
    train_labels_encoded = le.transform(train_labels)
    le.fit(test_labels)
    test_labels_encoded = le.transform(test_labels)

    x_train_feature = pre_train
    x_test_feature = pre_valid
    y_train = train_labels_encoded
    y_test = test_labels_encoded

    print('******************************************************************** ')
    print('Feature extraction finished')
    print('******************************************************************** ')

    return x_train_feature, y_train, x_test_feature, y_test
