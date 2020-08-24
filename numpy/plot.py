import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt

RNN = glob.glob("./RNN/*")

timeframe =['2011/01', '2011/02', '2011/03', '2011/04', '2011/05', '2011/06',
    '2011/07', '2011/08', '2011/09', '2011/10', '2011/11', '2011/12', 
    '2012/01', '2012/02', '2012/03', '2012/04', '2012/05', '2012/06',
    '2012/07', '2012/08', '2012/09', '2012/10', '2012/11', '2012/12',
    '2013/01', '2013/02', '2013/03', '2013/04', '2013/05', '2013/06',
    '2013/07', '2013/08', '2013/09', '2013/10', '2013/11', '2013/12',
    '2014/01', '2014/02', '2014/03', '2014/04', '2014/05', '2014/06', 
    '2014/07', '2014/08', '2014/09', '2014/10', '2014/11', '2014/12', 
    '2015/01', '2015/02', '2015/03', '2015/04', '2015/05', '2015/06', 
    '2015/07', '2015/08', '2015/09', '2015/10', '2015/11', '2015/12',
    '2016/01', '2016/02', '2016/03', '2016/04', '2016/05', '2016/06',
    '2016/07', '2016/08', '2016/09', '2016/10', '2016/11', '2016/12',
    '2017/01', '2017/02', '2017/03', '2017/04', '2017/05', '2017/06',
    '2017/07', '2017/08', '2017/09', '2017/10', '2017/11', '2017/12',
    '2018/01', '2018/02', '2018/03', '2018/04', '2018/05', '2018/06',
    '2018/07', '2018/08', '2018/09', '2018/10', '2018/11', '2018/12',
    '2019/01','2019/02', '2019/03', '2019/04', '2019/05', '2019/06',
    '2019/07','2019/08', '2019/09', '2019/10', '2019/11', '2019/12', 
    '2020/01','2020/02', '2020/03', '2020/04']

for row in RNN:
    try:
        if  "prediction" in row:
            name = row.replace("./RNN","").replace("prediction","").replace("npy","")

            real = np.load(row.replace("prediction","label"))
            rnn = data = np.load(row)
            lstm = data = np.load(row.replace("RNN","LSTM"))
            fuzzy_rnn = np.load(row.replace("RNN","fuzzy_RNN"))
            fuzzy_lstm = data = np.load(row.replace("RNN","anfis"))
            anfis = data = np.load(row.replace("RNN","LSTM"))
            dr = data = np.load(row.replace("RNN","DR"))

            fig, ax1 = plt.subplots(1,1,figsize=(20,10))
            ax1.plot(timeframe[-10:],real[-10:], label='Actual')
            ax1.plot(timeframe[-10:],rnn[-10:], label='RNN', marker='.')
            ax1.plot(timeframe[-10:],lstm[-10:], label='LSTM', marker='p')
            ax1.plot(timeframe[-10:],fuzzy_rnn[-10:], label='Fuzzy RNN', marker='D')
            ax1.plot(timeframe[-10:],fuzzy_lstm[-10:], label='Fuzzy LSTM', marker='d' )
            ax1.plot(timeframe[-10:],dr[-10:], label='Decision tree Regression', marker='h')
            ax1.plot(timeframe[-10:],anfis[-10:], label='ANFIS', marker='o')
            # ax1.annotate(ano, 
            #      xy=(0.75, 0.9),  xycoords='axes fraction',
            #     xytext=(0.75, 0.9), textcoords='axes fraction')
            ax1.set_title("Chi-a Prediction at "+name,fontsize=13)
            ax1.legend()

            fig.autofmt_xdate()
            ax1.set_ylim(bottom=0)
            ax1.set_ylim(top=100)

            # plt.show()
            plt.savefig("final/"+ name +'.png', dpi=100)
    except:
        continue