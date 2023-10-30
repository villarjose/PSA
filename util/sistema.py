from util.modelo_mecanicista import PredecirPSAOneDosis
import pandas as pd
import numpy as np

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
