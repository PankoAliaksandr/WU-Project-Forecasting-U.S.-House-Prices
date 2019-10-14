# Libraries
import pandas as pd
from matplotlib import pyplot
from pandas_datareader import data as pdr
import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics import tsaplots
from statsmodels.tsa.arima_model import ARIMA


class Home_Price_Index:

    def __init__(self):
        start_date = datetime.datetime(1975, 1, 1)
        end_date = datetime.date.today()
        self.__orig_data = pdr.DataReader('CSUSHPISA', 'fred', start_date,
                                          end_date)
        self.__orig_data = self.__orig_data['CSUSHPISA']
        self.__data = self.__orig_data
        self.__ADF_test = list()
        self.__predictions_ARIMA = None
        self.__forecasted = None

    def __execute_ADF_test(self):
        self.__ADF_test = adfuller(self.__data)
        print('ADF Statistic: %f' % self.__ADF_test[0])
        print('p-value: %f' % self.__ADF_test[1])

    def __make_stationary(self):
        self.__data = self.__data - self.__data.shift()
        self.__data = self.__data.dropna()
        self.__execute_ADF_test()
        self.__data.plot()

    def __implement_ARIMA(self):
            # ARIMA(2,0 ,0)
            model = ARIMA(self.__data, order=(2, 0, 0))
            model_fit = model.fit(disp=0)
            print(model_fit.summary())
            pyplot.plot(self.__data)
            pyplot.plot(model_fit.fittedvalues, color='red')
            pyplot.title('Model fit')
            pyplot.show()

            # Restore the data
            self.__predictions_ARIMA = pd.Series(model_fit.fittedvalues,
                                                 copy=True)

            self.__predictions_ARIMA = self.__predictions_ARIMA.cumsum() + \
                self.__orig_data[0]

            pyplot.plot(self.__orig_data, label='Original index values')
            pyplot.plot(self.__predictions_ARIMA, color='red',
                        label='Predicted index values')

            pyplot.legend()
            # Forcast difference
            self.__forecasted = model_fit.forecast(steps=3)[0]
            # Restore
            self.__forecasted = self.__forecasted.cumsum() + \
                self.__orig_data[len(self.__orig_data)-1]

            pyplot.show()

    def get_ARIMA_forcast(self):
        print self.__forecasted
        return self.__forecasted

    def main(self):
        self.__data.plot()
        self.__execute_ADF_test()
        self.__make_stationary()
        tsaplots.plot_acf(self.__data, lags=30)
        pyplot.show()
        tsaplots.plot_pacf(self.__data, lags=50)
        pyplot.show()
        self.__implement_ARIMA()
        self.get_ARIMA_forcast()


HPI = Home_Price_Index()
HPI.main()
