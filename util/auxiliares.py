import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.manifold import TSNE
from scipy.stats import ttest_ind

def cargar_datos(ruta: str, hoja: list = ['OneDose', 'Relapse', 'times']):

    patient_data = pd.read_excel(ruta, sheet_name=hoja[0])
    relapse_data = pd.read_excel(ruta, sheet_name='Relapse')
    time_data = pd.read_excel(ruta, sheet_name='times')

    return(patient_data, relapse_data, time_data)

def dibujarPlotly(labels_pred,data,labels_real,titulo=None):
    labels = [str(x) for x in labels_pred]
    df = pd.DataFrame({'x':data[:,0], 'y':data[:,1], 'cluster':labels, 'recaida':labels_real})
    if labels_pred[0] == 1:
        fig = px.scatter(df, x='x', y='y', color='cluster', symbol = 'recaida', 
                     symbol_sequence= ['circle','x'], 
                     color_discrete_sequence = ['blue', 'orange', 'green', 'brown', 'red', 'purple', 'black'],
                     width=600, height=400,
                     hover_data=[df.index])
    else:
        fig = px.scatter(df, x='x', y='y', color='cluster', symbol = 'recaida', 
                     symbol_sequence= ['x', 'circle'], 
                     color_discrete_sequence = ['blue', 'orange', 'green', 'brown', 'red', 'purple', 'black'],
                     width=600, height=400,
                     hover_data=[df.index])

    fig.update_layout(title=titulo)
    fig.show()

def ttest_df(df_1,df_2,columna):
    t_stat, p_value = ttest_ind(np.array(df_1.loc[:,columna]), np.array(df_2.loc[:,columna]))
    print("T-statistic value: ", t_stat)
    print("P-Value: ", p_value)
    alpha = 0.05
    if p_value < alpha:
        print("There is a statistically significant difference between the means.")
    else:
        print("There is no statistically significant difference between the means.")

def ttest_df_soloSignificante(df_1,df_2,columna):
    t_stat, p_value = ttest_ind(np.array(df_1.loc[:,columna]), np.array(df_2.loc[:,columna]))
    # print("T-statistic value: ", t_stat)
    # print("P-Value: ", p_value)
    alpha = 0.05
    if p_value < alpha:
        print(columna,':')
        print("There is a statistically significant difference between the means. P-Value: ", p_value)
    # else:
    #     print("There is no statistically significant difference between the means.")


def tsne_graph(X,y,dibujar=True, perplexity=20, lr='auto'):
    sistema = TSNE(n_components=2, perplexity=perplexity, learning_rate=lr, random_state=0)
    X_tsne = sistema.fit_transform(X)
    fig = None
    if dibujar == True:
        fig = px.scatter(pd.DataFrame(X_tsne), x = 0, y = 1, color=np.array([str(x) for x in y]), color_discrete_map={'0': 'slateblue', '1': 'orangered'})
        fig.show()
    return X_tsne, fig

def recortar_array(arr, indices_a_eliminar):
    if type(arr) == type(np.array(0)):
        return np.delete(arr, indices_a_eliminar, axis=0) 
    elif str(type(arr)) == "<class 'pandas.core.frame.DataFrame'>":
        return arr.drop(indices_a_eliminar, axis=0)
    elif str(type(arr)) == "<class 'list'>":
        arr = np.ndarray(arr)
        return np.delete(arr, indices_a_eliminar, axis=0)
    else:
        raise Exception('No se ha definido operación para este tipo de dato:',type(arr))



import plotly.graph_objects as go
from plotly.subplots import make_subplots

def dibujar_histogramas_por_columna(df):
    # Especifica el número de columnas por fila
    columnas_por_fila = 3
    ancho_figura = 1000  # Ancho de la figura en píxeles
    alto_figura = 1200  # Alto de la figura en píxeles
    ancho_subplot = ancho_figura // columnas_por_fila  # Ancho de cada subplot

    # Inicializa una figura con subplots
    fig = make_subplots(rows=(len(df.columns) // columnas_por_fila) + 1, cols=columnas_por_fila)

    # Crea un histograma para cada columna
    for i, columna in enumerate(df.columns, 1):
        row = (i - 1) // columnas_por_fila + 1
        col = (i - 1) % columnas_por_fila + 1

        # Agrega un histograma a la figura
        fig.add_trace(go.Histogram(x=df[columna], showlegend=False), row=row, col=col)
        fig.update_xaxes(title_text=columna, row=row, col=col, title_standoff=0)
        fig.update_yaxes(title_text='Frecuencia', row=row, col=col, title_standoff=0)

    # Actualiza el tamaño de la figura
    fig.update_layout(
        width=ancho_figura,
        height=alto_figura,
        title_text="Histograma para cada columna de X"
    )

    # Muestra la figura
    fig.show()