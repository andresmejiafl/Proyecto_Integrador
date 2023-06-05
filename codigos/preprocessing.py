import glob
import cv2
import os

def preprocesamiento_imagenes(path):
    """
    Función para el preprocesamiento de las imagenes, comvietiendolo a escala de grises y ecualizando el brillo.
    Tiene como unico parametro de entrada la ruta con las imagenes de train y test 
    previamente separadas.
    """
    print('******************************************************************** ')
    print('Cargando imagenes')
    print('******************************************************************** ')

    # Lista para almacenar las imágenes cargadas

    train_images = []
    test_images = []
    train_labels = []
    test_labels = []

    for directory_path in glob.glob(path + "/train/*"):

        label_tr = directory_path.split("\\")[-1]

        for img_path in glob.glob(os.path.join(directory_path, "*.png")):

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            train_images.append(img)
            train_labels.append(label_tr)

    print(f"Se cargaron {len(train_images)} imágenes de train.")
 
    for directory_path in glob.glob(path + "/test/*"):

        label_te = directory_path.split("\\")[-1]

        for img_path in glob.glob(os.path.join(directory_path, "*.png")):

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            test_images.append(img)
            test_labels.append(label_te)

    print(f"Se cargaron {len(test_images)} imágenes de test.")
    
    for i in range(len(train_images)):
        imagen=cv2.cvtColor(train_images[i], cv2.COLOR_BGR2GRAY)
        imagen_ecualizada = cv2.equalizeHist(imagen)
        cv2.imwrite(path + f"gray/train/{train_labels[i]}_gray_{i}.png",imagen)
        cv2.imwrite(path + f"equalizer/train/{train_labels[i]}_equa_{i}.png",imagen_ecualizada)

    for i in range(len(test_images)):
        imagen=cv2.cvtColor(test_images[i], cv2.COLOR_BGR2GRAY)
        imagen_ecualizada = cv2.equalizeHist(imagen)
        cv2.imwrite(path + f"gray/test/{test_labels[i]}_gray_{i}.png",imagen)
        cv2.imwrite(path + f"equalizer/test/{test_labels[i]}_equa_{i}.png",imagen_ecualizada)
    
    print('******************************************************************** ')
    print('Carga de imagenes finalizada')
    print('******************************************************************** ')