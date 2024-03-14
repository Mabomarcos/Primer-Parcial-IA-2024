#MARCOS ANTONIO BENITEZ OCAMPOS 4.661.558
import pandas as pd
import numpy as np
# Función para calcular los coeficientes de regresión manualmente
def regresion_manual(X, y):
    # Agregar una columna de unos para el término independiente
    X =np.column_stack([np.ones(len(X)), X])
    # Calcular los coeficientes utilizando la fórmula de la pseudo inversa
    coeficientes =np.linalg.pinv(X.T@X)@X.T@y
    return coeficientes  
# Función para predecir los valores de y
def predecir(X, coeficientes):
    Xm=np.column_stack([np.ones(len(X)), X])
    return  Xm@coeficientes#@ es multiplicacion de matrices
# Calcular métricas de evaluación manualmente
def rmse(y_true, y_pred):
    error=y_true-y_pred
    return np.sqrt(np.mean((error) ** 2))
def r2F(y_true, y_pred):
    numerador = ((y_true - y_pred) ** 2).sum()
    denominador = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - (numerador / denominador)

# Función para ajustar el modelo y evaluarlo
def ajustar_evaluar_modelo(X, y):
    coeficientes = regresion_manual(X, y)
    y_pred = predecir(X, coeficientes)
    r2_ =r2F(y, y_pred)
    rmse_val =rmse(y, y_pred)
    return coeficientes, y_pred, r2_, rmse_val
opcion=int(input())
# Cargar los datos
#mediciones=pd.read_excel('Mediciones22-02-21.xlsx')
#mediciones.head()
data=pd.read_csv('Mediciones.csv')
# Definir las columnas de características (X) y la columna de objetivo (y)
if opcion==1:
    #imprimir numero de filas y numero de columnas
    print("Número de filas y columnas:", data.shape)
    #seleccionar las caracteristicas(variables dependientes) y el objetivo
    caracteristicas =['PEEP','BPM','VTI','VTE','VTI_F','VTE_F']
    objetivo =['Pasos']
    X = data[caracteristicas]
    y = data[objetivo]
    print("Características:")
    print(X.head())
    print("Objetivo:")
    print(y.head())
elif opcion==2: 
    # modelo completo solo con VTI_F, completar la funcion regresion manual
    X = data['VTI_F']
    y = data['Pasos']
    coef=regresion_manual(X, y)
    print("Coeficientes del modelo VTI_F:",coef)
elif opcion==3: 
    # modelo completo solo con VTI_F, completar las funciones que definen las métricas
    X = data['VTI_F']
    y = data['Pasos']
    coef = regresion_manual(X, y)
    print( coef)
    y_pred = predecir(X,coef)
    r2_ = r2F(y, y_pred)
    rmse_val = rmse(y, y_pred)
    # imprimir los primeros 2 elementos de y e y_pred
    #  print(y[:3],  y_pred [COMPLETAR])4
    print(y[:3],  y_pred [:3])
    # imprimir r2 y rmse
    print(r2_,  rmse_val )
elif opcion==4: 
    # modelo completo solo con VTI_F, completar la función ajustar_evaluar_modelo
    X_todo =data['VTI_F']  #data[completar]
    y =data['Pasos'] # data[completar]
    coeficientes_todo, y_pred_todo, r2_todo, rmse_todo = ajustar_evaluar_modelo(X_todo, y)
    print(r2_todo, rmse_todo)
elif opcion==5:
   # Completar la combinaciones de características de los modelos solicitados 
    models = {
        'Modelo_1':['VTI_F'],
        'Modelo_2':['VTI_F','BPM'],
        'Modelo_3':['VTI_F','PEEP'],
        'Modelo_4':['VTI_F','PEEP','BPM'],
        'Modelo_5':['VTI_F','PEEP','BPM','VTE_F']
    }
    for nombre_modelo, lista_caracteristicas in models.items():
        X =data[lista_caracteristicas]
        y = data['Pasos']
        coeficientes, y_pred, r2, rmse_val = ajustar_evaluar_modelo(X, y)
        print(nombre_modelo,r2, rmse_val) 
elif opcion==6:
    # Modelos para cada combinación de PEEP y BPM
    valores_peep_unicos =data['PEEP'].unique()#completar sugerencia, utilizar unique()
    valores_bpm_unicos =data['BPM'].unique() #completar
    print(valores_peep_unicos)
    print(valores_bpm_unicos)
    predicciones_totales = []
    for peep in valores_peep_unicos:
        for bpm in valores_bpm_unicos:
            datos_subset = data[(data['PEEP'] == peep) & (data['BPM'] == bpm)] #completar el filtrado de datos, se deben filtrar los datos para cada para par de PEEP y BPM
            X_subset = datos_subset[['VTI_F']]
            y_subset = datos_subset['Pasos']
            coeficientes_subset, y_pred_subset, r2_subset, rmse_subset = ajustar_evaluar_modelo(X_subset, y_subset)
            print(peep, bpm, r2_subset, rmse_subset)
            predicciones_totales.append(y_pred_subset)
    predicciones_concatenadas = np.concatenate(predicciones_totales)
    y=data['Pasos']
    r2_global = r2F(y, predicciones_concatenadas)
    rmse_global = rmse(y, predicciones_concatenadas)
    print('Global', r2_global, rmse_global)