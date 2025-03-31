# Вынесем всю функциональность регрессионной модели в обособленный класс

from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

train_data: np.ndarray = pd.read_csv("CSV/DailyDelhiClimateTrain.csv").values
test_data: np.ndarray = pd.read_csv("CSV/DailyDelhiClimateTest.csv").values

class PogodaPrediction:
    class Normalize:
        def __init__(self, data: np.ndarray):
            self.data = np.copy(data)
            self.__mean = data.mean(axis=0)
            self.__std_dev = data.std(axis=0)

        def normalizeData(self) -> np.ndarray:
            return (self.data - self.__mean) / self.__std_dev

        def DeNormalizeData(self, normalized_data: np.ndarray, axes: list[int] = None) -> np.ndarray:
            if axes is None:
                axes = list(range(self.__mean.shape[0]))
            return normalized_data * self.__std_dev[axes] + self.__mean[axes]

    class AdamOptimizer: 
        def __init__(self, parameters_shape: tuple, speed: float = 0.001, 
                     beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
            self.speed = speed
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.iteration_step = 0

            self.m_parameters = np.zeros(parameters_shape)
            self.v_parameters = np.zeros(parameters_shape)
            self.m_bias = 0.0
            self.v_bias = 0.0

        def step(self, parameters: np.ndarray, bias: float, 
                 gradient_parameters: np.ndarray, gradient_bias: float) -> tuple:
            self.iteration_step += 1

            self.m_parameters = self.beta1 * self.m_parameters + (1 - self.beta1) * gradient_parameters
            self.v_parameters = self.beta2 * self.v_parameters + (1 - self.beta2) * (gradient_parameters ** 2)
            
            m_hat_parameters = self.m_parameters / (1 - self.beta1 ** self.iteration_step)
            v_hat_parameters = self.v_parameters / (1 - self.beta2 ** self.iteration_step)
            
            parameters -= self.speed * m_hat_parameters / (np.sqrt(v_hat_parameters) + self.epsilon)

            self.m_bias = self.beta1 * self.m_bias + (1 - self.beta1) * gradient_bias
            self.v_bias = self.beta2 * self.v_bias + (1 - self.beta2) * (gradient_bias ** 2)
            
            m_hat_bias = self.m_bias / (1 - self.beta1 ** self.iteration_step)
            v_hat_bias = self.v_bias / (1 - self.beta2 ** self.iteration_step)
            
            bias -= self.speed * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)
            return parameters, bias

    def __init__(self, test_data: np.ndarray, train_data: np.ndarray):
        self.epochs = 3000
        self.learning_rate = 0.001
        self.noise_std = 0.05
        self.number_of_sinuses = 10

        self.test_data = test_data.copy()
        self.train_data = train_data.copy()
        self._process_data()

    def _process_data(self):
        self._transform_dates_to_correct_form()
        
        self.train_normalize_class = self.Normalize(self.train_data[:, 1:])
        self.train_data[:, 1:] = self.train_normalize_class.normalizeData()
        
        self._transform_data_with_SMA()
        
        self._prepare_parameters()

    def _transform_dates_to_correct_form(self):
        self.train_data[:, 0] = np.vectorize(self.days_since_zero_date)(self.train_data[:, 0])
        self.test_data[:, 0] = np.vectorize(self.days_since_zero_date)(self.test_data[:, 0])
        self.train_data = self.train_data.astype(np.float64)
        self.test_data = self.test_data.astype(np.float64)

    def _transform_data_with_SMA(self):
        window_size = max(1, self.train_data.shape[0] // 70)
        number_of_columns = self.train_data.shape[1] - 1
        
        number_of_lines = self.train_data.shape[0] - window_size + 1
        self.denoised_data = np.zeros((number_of_lines, number_of_columns))
        
        for i in range(number_of_columns):
            self.denoised_data[:, i] = self.SMA(self.train_data[:, i + 1], window_size)
        
        self.x_data = np.arange(number_of_lines)

    def _prepare_parameters(self):
        number_of_columns = self.denoised_data.shape[1]
        self.mfft = []
        self.imax = []
        self.init_params = []
        self.bias = []
        
        for i in range(number_of_columns):
            fft = np.fft.fft(self.denoised_data[:, i])
            self.mfft.append(fft)
            
            imax = np.argsort(np.abs(fft))[::-1][:self.number_of_sinuses]
            self.imax.append(imax)
            
            freqs = imax / len(self.denoised_data[:, i])
            parameters = np.array([
                [np.std(self.denoised_data[:, i]), freq * 2 * np.pi, 0] 
                for freq in freqs
            ])
            self.init_params.append(parameters)
            self.bias.append(np.mean(self.denoised_data[:, i]))

    @staticmethod
    def days_since_zero_date(date_str: str) -> int:
        date_format: str = "%Y-%m-%d"
        date_obj = datetime.strptime(date_str, date_format)
        zero_date = datetime.strptime("2013-01-01", date_format)
        delta = date_obj - zero_date
        return delta.days

    @staticmethod
    def SMA(data: np.ndarray, window: int) -> np.ndarray:
        sma_values = np.zeros(len(data) - window + 1)
        for i in range(len(sma_values)):
            sma_values[i] = np.mean(data[i : i + window])
        return sma_values

    def forward(self, parameters: np.ndarray, bias: float, time: np.ndarray) -> np.ndarray:
        y_predicted = np.full_like(time, bias, dtype=np.float64)
        for A, omega, phi in parameters:
            y_predicted += A * np.sin(omega * time + phi)
        return y_predicted

    def train(self, column: int):
        if column >= self.denoised_data.shape[1]:
            print("Некорректный номер столбца")
            return
            
        noise = np.random.normal(0, self.noise_std, self.denoised_data.shape[0])
        y_noisy = self.denoised_data[:, column] + noise
        
        optimizer = self.AdamOptimizer(
            parameters_shape=self.init_params[column].shape,
            speed=self.learning_rate
        )
        
        parameters = self.init_params[column].copy()
        bias = self.bias[column]
        for epoch in range(self.epochs):
            grad_params, grad_bias = self._compute_gradients(parameters, bias, y_noisy)
            parameters, bias = optimizer.step(parameters, bias, grad_params, grad_bias)
            
            if epoch % 100 == 0:
                loss = self._compute_MSE_loss(parameters, bias, y_noisy)
                print(f"Epoch {epoch}: Loss={loss:.4f}")
        
        self.bias[column] = bias
        self.init_params[column] = parameters
        self.predictions_noisy = self.forward(parameters, bias, self.x_data) + np.random.normal(0, self.noise_std, self.x_data.size)

    def _compute_gradients(self, parameters: np.ndarray, bias: float, y_true: np.ndarray):
        y_predicted = self.forward(parameters, bias, self.x_data)
        error = y_predicted - y_true
        n = len(y_true)
        
        grad_params = np.zeros_like(parameters)
        for i, (A, omega, phi) in enumerate(parameters):
            t = omega * self.x_data + phi
            grad_params[i, 0] = 2/n * np.sum(error * np.sin(t))
            grad_params[i, 1] = 2/n * np.sum(error * A * self.x_data * np.cos(t))
            grad_params[i, 2] = 2/n * np.sum(error * A * np.cos(t))
            
        grad_bias = 2/n * np.sum(error)
        return grad_params, grad_bias

    def _compute_MSE_loss(self, parameters: np.ndarray, bias: float, y_true: np.ndarray) -> float:
        y_predicted = self.forward(parameters, bias, self.x_data)
        return np.mean((y_predicted - y_true)**2)

    def _compute_MAE_loss(self, y_predicted: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean(np.abs(y_predicted - y_true))

    def visualize_results(self, column: int):
        if column >= self.denoised_data.shape[1]:
            print("Некорректный номер столбца")
            return

        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.denoised_data[:, column], label='Исходные данные')
        plt.plot(self.predictions_noisy, label='Прогноз', alpha=0.7)
        plt.title("Сравнение с исходными данными")
        plt.legend()
        
        denormalized_predictions = self.train_normalize_class.DeNormalizeData(
            self.predictions_noisy, 
            axes=[column]
        )[:len(self.test_data)]

        plt.subplot(2, 1, 2)
        plt.plot(self.test_data[:, column + 1], label='Тестовые данные')
        plt.plot(denormalized_predictions, label='Денормализованный прогноз')
        plt.title("Сравнение с тестовыми данными")
        plt.legend()
        
        print(self._compute_MAE_loss(denormalized_predictions, self.test_data[:, column]))
        plt.show()

model = PogodaPrediction(test_data, train_data)
for i in range(4):
    model.train(i)
    model.visualize_results(i)