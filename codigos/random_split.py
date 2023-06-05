from shutil import copyfile
import os
import random

def random_split(dir_img,dir_train,dir_test,train_size):
    """
    Esta función divide la base de datos de imagenes en
    train y test de forma aleatoria, separandola por carpetas.
    Tiene como parametros las rutas de los directorios de las
    imagenes, directorio train, directorio test y porcentaje 
    de entrenamiento.    
    """
    print('******************************************************************** ')
    print('Iniciando división del conjunto de datos')
    print('******************************************************************** ')

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
    
    print('******************************************************************** ')
    print('Finalizando división del conjunto de datos')
    print('******************************************************************** ')
