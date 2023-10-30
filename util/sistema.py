from util.modelo_mecanicista import PredecirPSAOneDosis
import pandas as pd
import numpy as np
import optuna
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
from sklearn.metrics import v_measure_score


class Sistema(PredecirPSAOneDosis):

    def __init__(self) -> None:
        pass

    def cargar_datos(self, ruta, hojas):
        """
            Se cargan múltiples hojas de un mismo arhivo de excel y se devuelven en formato de lista
        """        
        datos = []
        for hoja in hojas:
            datos.append(pd.read_excel(ruta, sheet_name=hoja))

        return datos

    def cargar_ics(self, ruta, size):
        ics = pd.DataFrame(pd.read_excel(ruta, sheet_name=str(1))).transpose()
        for sheet_name in range(1, size):
            sheet_data = pd.read_excel(ruta, sheet_name=str(sheet_name)).transpose()
            ics = pd.concat([ics, sheet_data], axis=0, ignore_index=True)
        return ics

    def crear_datasets(self, ruta, ruta_ics, t_n):
        """
            Parameters
            ----------
            ruta : str
                Ruta del archivo de EXCEL con los datos de los pacientes.
                Debe contener las siguientes hojas:
                    - hoja 1: datos de PSA por paciente
                    - hoja 2: clase de cada paciente
                    - hoja 3: tiempo de cada medida de PSA por paciente
                    - hoja 4: edad de cada paciente
                    - hoja 5: gleason_scores de cada paciente
            ruta_ics : str
                Ruta del archivo de EXCEL con los datos de los Intervalos de Confianza (I.C.) 
                de los pacientes.
            t_n : int
                Último instante de tiempo de medida de PSA que estamos teniendo en cuenta 
                para ajustar los valores.

            Returns
            ------
            Se devuelven (como tupla):
                - Un dataframe con los valores X de cada paciente.
                - Un vector con las clases de cada paciente.
                - Un dataframe con los tiempos de cada paciente.
        """

        hojas = ['parametros', 'relapse', 'times', 'edad', 'gs']
        df_parametros, y, df_tiempos, df_edad, df_gs = self.cargar_datos(ruta, hojas)

        filtro = df_parametros['Case'] == "OneDose"
        df_parametros = df_parametros[filtro].reset_index(drop=True)

        predictor = PredecirPSAOneDosis(df_parametros, y, df_tiempos)
        
        lista_t_n       = [predictor.psa_paciente_instante_t(int(x),f"t_{t_n}") for x in range(0,df_parametros.shape[0])]
        t_n_mas_1 = f"t_{t_n+1}"
        t_n_mas_2 = f"t_{t_n+2}"
        lista_t_n_mas_1 = [predictor.psa_paciente_instante_t(int(x),t_n_mas_1) for x in range(0,df_parametros.shape[0])]
        lista_t_n_mas_2 = [predictor.psa_paciente_instante_t(int(x),t_n_mas_2) for x in range(0,df_parametros.shape[0])]
        lista_psa_nadir = [predictor.psa_nadir_paciente(paciente) for paciente in range(0,df_parametros.shape[0])]
        lista_ics = self.cargar_ics(ruta_ics, df_parametros.shape[0])
        lista_ic_t_n = np.array(lista_ics.iloc[:,-1])

        lista_edades = df_edad.loc[:df_parametros.shape[0],'age_ebrt'].values
        lista_gg1 = df_gs.loc[0:df_parametros.shape[0],'GG1'].values
        lista_gg2 = df_gs.loc[0:df_parametros.shape[0],'GG2'].values
        lista_gs  = df_gs.loc[0:df_parametros.shape[0],'GS'].values

        # df_params = df_parametros.iloc[:,2:11]

        df = pd.DataFrame({'PSA_0':df_parametros.loc[:,'P0'],
                           'R':df_parametros.loc[:,'R'],
                           'rho_d':df_parametros.loc[:,'rho_d'],
                           'rho_s':df_parametros.loc[:,'rho_s'],
                           'beta':df_parametros.loc[:,'beta'],
                           'alpha':df_parametros.loc[:,'alpha'],
                           'T_n':df_parametros.loc[:,'T_n'],
                           'P_n':df_parametros.loc[:,'P_n'],
                           'Delta_T_n':df_parametros.loc[:,'Delta_T_n'],
                           f't_{t_n}':lista_t_n,
                           'ic_t_n':lista_ic_t_n,
                           t_n_mas_1:lista_t_n_mas_1,
                           t_n_mas_2:lista_t_n_mas_2,
                           'psa_nadir':lista_psa_nadir,
                           'edad':lista_edades,
                           'gg1':lista_gg1,
                           'gg2':lista_gg2})
        
        return df, y, df_tiempos


    def tsne_spectralclustering(self, X, p, n):
        tsne = TSNE(n_components=2, random_state=0, perplexity=p)
        clusterizado = SpectralClustering(n_clusters=n)
        X_tsne = tsne.fit_transform(self.X)
        y_clust = clusterizado.fit_predict(X_tsne)
        return X_tsne, y_clust
    
    def optuna_tsne_spectralclustering(self, trial):
        tsne_p  = trial.suggest_int("tsne_p", 10, 100)
        clust_n = trial.suggest_int("clust_n", 2, 10)
        _, y_clust = self.tsne_spectralclustering(self.X, tsne_p,clust_n)
        return v_measure_score(self.y_optuna.iloc[:,1], y_clust)

    def optimizar_tsne_spectralclustering(self, X, y):
        self.X_optuna = X.copy()
        self.y_optuna = y.copy()

        study = optuna.create_study(direction="maximize")
        study.optimize(self.optuna_tsne_spectralclustering, n_trials=150, show_progress_bar=True, timeout=300) #show_progress_bar=True  #n_jobs=-1, 

        return study.best_params, study.best_value, study.best_trial
    
