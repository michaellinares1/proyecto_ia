# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoLars, BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from keras.models import Sequential
from keras.layers import Dense

class NeuralNetworkModel:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.model = None
    def build_model(self):
        self.model = Sequential([
            Dense(64, activation='relu', input_dim=self.X_train.shape[1]),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def train_model(self, epochs=50, batch_size=32):
        self.model.fit(self.X_train, self.Y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    def evaluate_model(self):
        mse = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        y_pred = self.model.predict(self.X_test)
        return mse, y_pred
    
    def plot_predictions(self, y_pred):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.X_test, self.Y_test, color='black', label='Datos de prueba')
        plt.scatter(self.X_test, y_pred, color='blue', label='Predicciones', alpha=0.5)
        plt.xlabel("X Test")
        plt.ylabel("Y Test / Predicciones")
        plt.legend()
        plt.show()




class DataProcessor:
    def __init__(self, file_path, encoding='latin1'):
        self.file_path = file_path
        self.encoding = encoding
        self.dataset = None
    
    def load_data(self):
        self.dataset = pd.read_csv(self.file_path, encoding=self.encoding)
    
    def clean_data(self):
        columns_to_drop = ['FECHA_CORTE', 'DEPARTAMENTO', 'DISTRITO', 'PROVINCIA', 'SECTOR_UBICACION', 
                           'NOM_CONTRIBUYENTE', 'GOBIERNO_LOCAL', 'UBICACION_PREDIO']
        self.dataset = self.dataset.drop(columns=columns_to_drop)
        self.dataset = self.dataset.dropna()
    
    def get_data_info(self):
        return self.dataset.info(), self.dataset.describe()

class DataVisualizer:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def plot_tributo_count(self):
        plt.figure(figsize=(12, 6))
        plt.xticks(fontsize=8)
        ax = sns.countplot(x="COD_TRIBUTO", data=self.dataset)
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=8, color='black', xytext=(0, 5),
                        textcoords='offset points')
        plt.xlabel('Código de Tributo', fontsize=12)
        plt.ylabel('Conteo', fontsize=12)
        plt.title('Conteo por Código de Tributo', fontsize=16)
        plt.show()
    
    def plot_importe_promedio_por_anio(self):
        promedios_por_año = self.dataset.groupby('ANIO_DEUDA')['IMPORTE_CALCULADO'].mean().reset_index()
        plt.figure(figsize=(12, 6))
        barplot = sns.barplot(x="ANIO_DEUDA", y="IMPORTE_CALCULADO", data=promedios_por_año)
        plt.xlabel('Año', fontsize=12)
        plt.ylabel('Promedio de Importe Calculado', fontsize=12)
        plt.title('Promedio de Importe Calculado por Año', fontsize=16)
        for index, value in enumerate(promedios_por_año['IMPORTE_CALCULADO']):
            barplot.text(index, value + 0.1, round(value, 2), ha='center', va='bottom', fontsize=10)
        plt.show()

class ModelTrainer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.X_train, self.X_test, self.Y_train, self.Y_test = None, None, None, None
    
    def split_data(self):
        X = self.dataset[['ANIO_DEUDA']]
        Y = self.dataset['IMPORTE_DEUDA']
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    def train_lasso_lars(self):
        lasso_lars = LassoLars()
        lasso_lars.fit(self.X_train, self.Y_train)
        y_pred_lasso = lasso_lars.predict(self.X_test)
        return lasso_lars, y_pred_lasso
    
    def train_bayesian_ridge(self):
        br = BayesianRidge()
        br.fit(self.X_train, self.Y_train)
        y_pred = br.predict(self.X_test)
        mse = mean_squared_error(self.Y_test, y_pred)
        r2 = r2_score(self.Y_test, y_pred)
        return br, y_pred, mse, r2

    def plot_model_predictions(self, model, y_pred):
        plt.scatter(self.X_test, self.Y_test, color='black', label='Datos de prueba')
        plt.plot(self.X_test, y_pred, color='blue', linewidth=3, label='Predicción')
        plt.title("Gráfico de Predicción vs Datos de Prueba")
        plt.xlabel("X Test")
        plt.ylabel("Y Test / Predicción")
        plt.legend()
        plt.show()

class QuantileRegression:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def perform_quantile_regression(self, quantile=0.5):
        data = pd.concat([self.dataset[['ANIO_DEUDA']], self.dataset['IMPORTE_DEUDA']], axis=1)
        model = smf.quantreg('IMPORTE_DEUDA ~ ANIO_DEUDA', data)
        res = model.fit(q=quantile)
        return res, quantile
    
    def plot_quantile_regression(self, res, quantile):
        x_pred = np.linspace(self.dataset['ANIO_DEUDA'].min(), self.dataset['ANIO_DEUDA'].max(), 100)
        y_pred = res.predict(pd.DataFrame({'ANIO_DEUDA': x_pred}))
        plt.scatter(self.dataset['ANIO_DEUDA'], self.dataset['IMPORTE_DEUDA'], alpha=0.5, label='Datos')
        plt.plot(x_pred, y_pred, color='red', label=f'Quantile {quantile}')
        plt.xlabel('ANIO_DEUDA')
        plt.ylabel('IMPORTE_DEUDA')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Procesamiento de datos
    file_path = 'DATASET_DEUDA_POR_COBRAR_PRINCIPALES_TRIBUTOS_MUNICIPALES_MPPAITA_2022_2023.csv'
    data_processor = DataProcessor(file_path)
    data_processor.load_data()
    data_processor.clean_data()
    data_processor.get_data_info()
    
    # Visualización de datos
    data_visualizer = DataVisualizer(data_processor.dataset)
    data_visualizer.plot_tributo_count()
    data_visualizer.plot_importe_promedio_por_anio()
    
    # Entrenamiento de modelos
    model_trainer = ModelTrainer(data_processor.dataset)
    model_trainer.split_data()
    lasso_model, y_pred_lasso = model_trainer.train_lasso_lars()
    model_trainer.plot_model_predictions(lasso_model, y_pred_lasso)
    
    bayesian_model, y_pred_br, mse, r2 = model_trainer.train_bayesian_ridge()
    model_trainer.plot_model_predictions(bayesian_model, y_pred_br)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    
    # Regresión cuantílica
    quantile_regression = QuantileRegression(data_processor.dataset)
    quantile_res, quantile = quantile_regression.perform_quantile_regression()
    print(quantile_res.summary())
    quantile_regression.plot_quantile_regression(quantile_res, quantile)

    
    # Red Neuronal con Keras
    nn_model = NeuralNetworkModel(model_trainer.X_train, model_trainer.Y_train, model_trainer.X_test, model_trainer.Y_test)
    nn_model.build_model()
    nn_model.train_model(epochs=50, batch_size=32)
    mse_nn, y_pred_nn = nn_model.evaluate_model()
    print(f"Mean Squared Error (Neural Network): {mse_nn}")
    nn_model.plot_predictions(y_pred_nn)