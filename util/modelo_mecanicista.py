import numpy as np
import pandas as pd

class PredecirPSAOneDosis:
    def __init__(self, X, y, times):
        self.X = X
        self.relapse = y
        self.time = times

# ## CALCULANDO PSA EN INSTANTE T
    def predecir_psa(self, t, t_D, P0, R, rho_d, rho_s):
        t_D = t_D / 30
        # rho_n = rho_s
        # theta_1 = 1
        # theta_2 = np.exp(t_D * (rho_s + rho_d))
        value = P0 * (R * np.exp(rho_s * t) + (1 - R) * np.exp(t_D * (rho_s + rho_d)) * np.exp(-rho_d * t))
        return value

    def psa_paciente(self, paciente_i, time):
        patient_row = self.X.iloc[paciente_i]
        t_D = self.time.loc[paciente_i, 't_1']
        return self.predecir_psa(time, t_D, patient_row['P0'], patient_row['R'], patient_row['rho_d'], patient_row['rho_s'])

    def psa_paciente_instante_t(self, paciente, instante): #self.df_times.columns[-1]
        x = int(self.time.loc[paciente,instante])
        devolver = self.psa_paciente(paciente, x/30)
        return devolver

# ## CALCULANDO PSA NADIR
    def predecir_psa_nadir(self, t_D, R, rho_d, rho_s):
        t_D = t_D / 30
        theta_2 = np.exp(t_D * (rho_s + rho_d))
        alpha = rho_s / rho_d
        beta = R / (1 - R)
        t_to_nadir = (1 / (1 + beta)) * np.log(theta_2 / (alpha * beta))
        psa_nadir = alpha * np.exp(beta * t_to_nadir)
        return psa_nadir
    
    def psa_nadir_paciente(self, paciente):
        t_D = self.time.loc[paciente, 't_1']
        return self.predecir_psa_nadir(t_D, self.X['R'][paciente], self.X['rho_d'][paciente], self.X['rho_s'][paciente])