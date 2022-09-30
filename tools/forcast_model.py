import os
import pandas as pd
import sys
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt

######### Global Variable  #############
STOCK_DATA_PATH = "../data/stock_data/"

class forcastModel:
    def __init__(self, df, pred_period=7):
        self.df = df
        self.pred_period = pred_period

    def model_prophet(self, display='plotly'):

        ## 포멧 변경하기
        df = self.df
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df = df[['Date', 'Close']]
        df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

        fbp = Prophet(daily_seasonality=True)

        # Fit the model
        fbp.fit(df)

        fut = fbp.make_future_dataframe(periods=self.pred_period)
        prediction = fbp.predict(fut)

        if display == 'plotly':
            fig =  plot_plotly(fbp, prediction)
            fig.show()
        else:
            fbp.plot(prediction)
            plt.show()

            fbp.plot_components(prediction)
            plt.show()






if __name__ == "__main__":

    ## 데이터 준비 확인

    try:
        file_list = os.listdir(STOCK_DATA_PATH)
        datapath = file_list[0]
    except Exception as e:
        print(e)



    df = pd.read_csv(STOCK_DATA_PATH + datapath)

    fm = forcastModel(df)
    fm.model_prophet()



