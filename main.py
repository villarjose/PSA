import pandas as pd
# import numpy as np
from util.sistema import Sistema
from util.auxiliares import dibujar_histogramas_por_columna
from sklearn.preprocessing import MinMaxScaler

log = True

def estandarizar(df):
    escalador = MinMaxScaler()
    datos_escalados = escalador.fit_transform(df)
    return(pd.DataFrame(datos_escalados,
                    index=df.index,
                    columns=df.columns))

def EDA(df):
    dibujar_histogramas_por_columna(df)
    print(df.shape)
    print(df.describe())
    print(estandarizar(df).head())


def main():
    sistema = Sistema()
    if log == True: print("Cargando datos...")
    X, y, tiempos = sistema.crear_datasets('datos/valores_10psas_166pacientes.xlsx', 'datos/ci_166pacientes_10psas.xlsx', 10) 

    if log == True: print("Explorando los datos...")
    EDA(X)



if __name__ == "__main__":
    main()


