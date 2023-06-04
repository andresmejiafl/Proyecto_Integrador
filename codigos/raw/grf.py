import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def graph_target(df):
    conteo = pd.value_counts(df["target"], sort = True)
    conteo.plot(kind = 'bar', rot=0)
    plt.xticks(range(4), ["Normal","Cataratas","Glaucoma","Retina"])
    plt.title("Distribución Clases")
    plt.xlabel("Etiqueta")
    plt.ylabel("Número de imagenes")
    plt.grid() 

def random_samples(df,path):
    images = df['source'].values
    random_images = [np.random.choice(images) for i in range(9)]
    plt.figure(figsize=(20,10))
    plt.suptitle("Muestra aleatoria de imagenes",fontsize=25)
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        img = plt.imread(os.path.join(path, random_images[i]))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()

def describe_image(df,path,flag_raw=True):
    if flag_raw==True:
        sample_img = df.source[0]
        raw_image = plt.imread(os.path.join(path, sample_img))
    else:
        raw_image = df[0]
    plt.imshow(raw_image, cmap='gray')
    plt.colorbar()
    plt.title('Imagen del fondo de ojo')
    print(f"Las dimensiones de la imagen son {raw_image.shape[0]} pixeles de alto y {raw_image.shape[1]} pixeles de ancho, un solo canal")
    print(f"El maximo valor de intensidad de un pixel es {raw_image.max():.4f} y el minimo valor de intensidad es {raw_image.min():.4f}")
    print(f"El valor promedio de la intesidad de los pixeles es {raw_image.mean():.4f} y la desviacion estandar de la intensidad es {raw_image.std():.4f}")

    sns.displot(raw_image.ravel(), 
                label=f'Promedio {np.mean(raw_image):.4f} & Desviacion estandar {np.std(raw_image):.4f}', 
                kde=False)
    plt.legend(loc='upper center')
    plt.title('Distribución de la intensidad de los pixeles de la imagen')
    plt.xlabel('Intensidad')
    plt.ylabel('# Pixeles en la imagen')
    plt.grid()

def raw_vs_std(df,path,df_std):
    sns.set()
    plt.figure(figsize=(10, 7))

    sample_img = df.source[0]
    raw_image = plt.imread(os.path.join(path, sample_img))

    sns.distplot(raw_image.ravel(), 
                label=f'Imagen original: Promedio {np.mean(raw_image):.4f} - Desviacion estandar {np.std(raw_image):.4f} \n '
                f'Valor minimo {np.min(raw_image):.4} - Valor maximo {np.max(raw_image):.4}',
                color='blue', 
                kde=False)

    sns.distplot(df_std[0].ravel(), 
                label=f'Imagen normalizada: Promedio {np.mean(df_std[0]):.4f} - Desviacion estandar {np.std(df_std[0]):.4f} \n'
                f'Valor minimo {np.min(df_std[0]):.4} - Valor maximo {np.max(df_std[0]):.4}', 
                color='red', 
                kde=False)

    plt.legend()
    plt.title('Distribución de intensidad de pixeles en la imagen')
    plt.xlabel('Intensidad')
    plt.ylabel('# Pixeles')

 