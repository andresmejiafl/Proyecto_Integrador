import os
import random
from shutil import copyfile

def random_split(dir_img,dir_train,dir_test,train_size):

    directorio_imagenes = dir_img
    proporcion_entrenamiento = train_size  
    proporcion_prueba = 1 - train_size  

    directorio_entrenamiento = dir_train
    directorio_prueba = dir_test

    os.makedirs(directorio_entrenamiento, exist_ok=True)
    os.makedirs(directorio_prueba, exist_ok=True)

    archivos_imagenes = os.listdir(directorio_imagenes)
    random.shuffle(archivos_imagenes)

    indice_corte = int(proporcion_entrenamiento * len(archivos_imagenes))

    for archivo in archivos_imagenes[:indice_corte]:
        origen = os.path.join(directorio_imagenes, archivo)
        destino = os.path.join(directorio_entrenamiento, archivo)
        copyfile(origen, destino)

    for archivo in archivos_imagenes[indice_corte:]:
        origen = os.path.join(directorio_imagenes, archivo)
        destino = os.path.join(directorio_prueba, archivo)
        copyfile(origen, destino)
    
    print("====================================================================")
