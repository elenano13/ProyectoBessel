import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import beta
from sklearn.metrics import mean_squared_error

#Definir la función beta
def beta_function(x, alpha, beta_param):
    return (x**(alpha - 1) * (1 - x)**(beta_param - 1)) / beta(alpha, beta_param)

#CSV cambiar según path
data_df = pd.read_csv(r'C:\Users\rodri\Downloads\Histogramas de la distribucion beta.csv')

#Extraer datos de los bins
bin_edges = data_df['Bin Edge'].values

#Se agrega un bin edge extra siguiendo la lógica del csv
bin_edges = np.append(bin_edges, 0.971576)

#Extraer datos de los histogramas
hist_data = [data_df[f'Histogram {i}'].values for i in range(1, 5)]

#Calcular bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

#Usar curve fit para cada histograma
for i, hist in enumerate(hist_data):
    
    plt.figure(figsize=(8, 6))
    
    #Densidad
    hist_density = hist / np.sum(hist) / np.diff(bin_edges)

    #Generar la fit curva
    initial_guess = [2, 2]  
    popt, _ = curve_fit(beta_function, bin_centers, hist_density, p0=initial_guess)
    alpha_opt, beta_param_opt = popt
    x_values = np.linspace(0, 1, 1000)  #por como se define beta [0,1]
    y_fit = beta_function(x_values, alpha_opt, beta_param_opt)

    #RMSE
    y_pred = beta_function(bin_centers, alpha_opt, beta_param_opt)
    rmse = np.sqrt(mean_squared_error(hist_density, y_pred))

    plt.bar(bin_edges[:-1], hist_density, width=np.diff(bin_edges), align='edge', alpha=0.6, color='purple', label='Histograma')
    plt.plot(x_values, y_fit, color='green', label=f'Curve fit\nα={alpha_opt:.2f}, β={beta_param_opt:.2f}\nRMSE={rmse:.4f}')

    plt.title(f'Histograma {i + 1}')
    plt.xlabel('x')
    plt.ylabel('Densidad')
    plt.legend()
    plt.grid(True)
    plt.show()
