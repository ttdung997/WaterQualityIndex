import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain, combinations
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import datetime
import pickle
import sys
from PIL import Image
import io

from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

# Thu vien math dung de goi cac ham tinh toan co ban (binh phuong, khai can)
from math import sqrt

from sklearn import linear_model
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

from matplotlib.dates import (YEARLY, DateFormatter,
                              rrulewrapper, RRuleLocator, drange)


from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# dataCol = ['Name (E)' ,'YY/MM','Dissolved Total N(㎎/L)', 'NH3-N(㎎/L)', 'NO3-N(㎎/L)', 'Dissolved Total P(㎎/L)','Conductivity(µS/㎝)','TSI(Chl-a)', 'Grade.3' ]

dataCol = ['Name (E)' ,'YY/MM','NH3-N(㎎/L)', 'NO3-N(㎎/L)', 'PO4-P(㎎/L)',
 'T-N(㎎/L)','T-P(㎎/L)', 'Dissolved Total N(㎎/L)','Dissolved Total P(㎎/L)',
  'Hydrogen ion conc.','DO (㎎/L)', 'TSI(Chl-a)']

dataCol = ['Name (E)' ,'YY/MM','Dissolved Total N(㎎/L)','Dissolved Total P(㎎/L)',
  'Hydrogen ion conc.','DO (㎎/L)','BOD(㎎/L)',"I(pH)", 'TSI(Chl-a)']

dataCol = ['Name (E)' ,'YY/MM','TSI(Chl-a)']

MasterDataframe = pd.read_excel('predata.xls')
MasterDataframe.rename(columns=MasterDataframe.iloc[0])
#Get all subset of the column
ColumnList = list(MasterDataframe)
# print(ColumnList)


# print(MasterDataframe.head())

MasterDataframe2 = pd.read_excel('Data.xlsx')
MasterDataframe2.rename(columns=MasterDataframe2.iloc[0])
#Get all subset of the column
# ColumnList = list(MasterDataframe)


df1 = MasterDataframe[dataCol]
df2 = MasterDataframe2[dataCol]

df = pd.concat([df1,df2])

# print(df.head())
# print(df.describe())

# print(df.isnull().sum())


for col in dataCol:
    try:
        # print(col)
        median = df[col].median()
        # print(median)
        # print("_____________________+")
        df[col].fillna(median, inplace=True)
    except:
        # print(col)
        # print("it here")
        df[col].fillna("Mesotrophic", inplace=True)

window_len = 5

myDf = df
# myDf = myDf.sort_values(by='YY/MM')

dictrictArr = (myDf['Name (E)'].unique())
# trash = ["Choicheon Stream","Gamicheon Stream",'Mangwolcheon Stream','Mokgamcheon Stream','Ahnyangcheon Stream 3-2','Min', 'Mesotrophic', 'Max' ,'Average']

# dictrictArr =[]
# for x in dictrictPreArr:
#   if x not  in trash:
#       dictrictArr.append(x)

training_input = np.empty((1,window_len,6))
training_output = np.empty((1,))
test_input = np.empty((1,window_len,6))
test_output = np.empty((1,))


# print(training_input)
# print(test_input)
# quit()
norm_cols = ['NH3-N(㎎/L)', 'NO3-N(㎎/L)', 'PO4-P(㎎/L)',
 'T-N(㎎/L)','T-P(㎎/L)', 'Dissolved Total N(㎎/L)','Dissolved Total P(㎎/L)',
  'Hydrogen ion conc.','DO (㎎/L)', 'TSI(Chl-a)']

dictrictLabel = []
dictrictValue = []


final_predict = []
true_label = []
# quit()
date_full =['2011/01', '2011/02', '2011/03', '2011/04', '2011/05', '2011/06',
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

def Average(lst): 
    return sum(lst) / len(lst) 

count = 0
lenValue = []

real_label = []
predict_label = []

finalMSE = []
finalDictrict = []

# print(dictrictArr)
# quit()
for dictrict in dictrictArr:
    # print(dictrict)
    count = count + 1
    small_data = df[df['Name (E)']==dictrict]
    small_data = small_data.drop('Name (E)', 1)
    # print(len(small_data))s
    if (len(small_data)) <= 48:
        continue

    dictrictLabel.append(dictrict)
    dictrictValue.append(np.array(small_data['TSI(Chl-a)'].values).tolist())

    # continue
    data_list = small_data.values
    dataDic = {}
    values_list = []
    for row in data_list:
        dataDic[row[0]] = row[1]
        values_list.append(row[1])

    for date in date_full:
        try:
            a =  dataDic[date]
        except:
            dataDic[date] = Average(values_list)
    RegressionData = []
    for key in dataDic:
        temp = []
        if int(key.split("/")[0]) > 2012:
            temp.append(dataDic[str(int(key.split("/")[0]) - 2)+"/"+key.split("/")[1]])
            temp.append(dataDic[str(int(key.split("/")[0]) - 1)+"/"+key.split("/")[1]])
            if int(key.split("/")[1]) == 1:
                temp.append(dataDic[str(int(key.split("/")[0]) - 1)+"/11"])
                temp.append(dataDic[str(int(key.split("/")[0]) - 1)+"/12"])
            elif int(key.split("/")[1]) == 2:
                temp.append(dataDic[str(int(key.split("/")[0]) - 1)+"/12"])
                temp.append(dataDic[(key.split("/")[0])+"/01"])
            elif int(key.split("/")[1]) == 12:
                temp.append(dataDic[key.split("/")[0]+"/10"])
                temp.append(dataDic[key.split("/")[0]+"/11"])
            elif int(key.split("/")[1]) == 11:
                temp.append(dataDic[key.split("/")[0]+"/09"])
                temp.append(dataDic[key.split("/")[0]+"/10"])
            else:
                temp.append(dataDic[(key.split("/")[0])+"/0"+str(int(key.split("/")[1])-2)])
                temp.append(dataDic[(key.split("/")[0])+"/0"+str(int(key.split("/")[1])-1)])
            temp.append(dataDic[key])
            RegressionData.append(temp)


    # np.savetxt("fdata/"+str(dictrict)+".txt",RegressionData,delimiter=' ' ,fmt='%1.4e') 
    # continue

    X = [x[:-1] for x in RegressionData]
    y = [x[-1] for x in RegressionData]

    # regr = linear_model.LinearRegression()
    # regr.fit(X[:62], y[:62])

    # regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    # regr = make_pipeline(StandardScaler(), SVR(C=100.0, epsilon=0.8))
    # regr.fit(X[:62], y[:62])
    # print(len(X))
    # continue

    regr = DecisionTreeRegressor(random_state=0)
    regr.fit(X[:62],y[:62])

    # cross_val_score(regr, X, y, cv=10)

    y_pred = regr.predict(X[62:])

    # Su dung metrics de tinh toan cac sai so cua mo hinh

    print("___________________________________________________________")
    # Sai so tuyet doi
    # print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred)) 

    # # Sai so binh phuong trung binh 
    # print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))  

    # # Sai so can bac 2 trung binh
    # print('Root Mean Squared Error:', sqrt(metrics.mean_squared_error(y, y_pred)))

    # # He so xac dinh : coefficient of determination (R2)
    # print('R^2 score:', (metrics.r2_score(y, y_pred)))


    # Sai so tuyet doi

    # for i in y:
    #   if i > 60:
    #       print("what")
    #       print(i)

    # print(y)
    # print(y_pred)
    # print("___")
    # print('Relative Mean Absolute Error:', metrics.mean_absolute_error(y, y_pred)/Average(y)) 

    # # Sai so binh phuong trung binh 
    # print(metrics.mean_absolute_error(y, y_pred))
    # print(max(y))




    finalMSE.append(metrics.mean_absolute_error(y[62:], y_pred)/Average(y))
    finalDictrict.append(dictrict)
    
    np.save('numpy/DR/'+ dictrict+"_label", y[62:])
    np.save('numpy/DR/'+ dictrict+"_prediction", y_pred)
    print('Relative Mean Squared Error:', metrics.mean_absolute_error(y[62:], y_pred)/Average(y))
    continue
    # real_label = [1 if i > 60 else 0 for i in y]
    # predict_label = [1 if i > 60 else 0 for i in y_pred]

    # from sklearn.metrics import classification_report
    # print(classification_report(real_label, predict_label))


    if y[-1] > 60 or y[-2] > 60 or y[-3] > 60 :
        real_label.append(1)
    else:
        real_label.append(0)

    if y_pred[-1] > 60 or y_pred[-2] > 60 or y_pred[-3] > 60 :
        predict_label.append(1)
    else:
        predict_label.append(0)



    # continue




    split_date = "2020/01"
    training_set, test_set = small_data[small_data['YY/MM']<"2019/09"], small_data[small_data['YY/MM']>="2019/09"]


    timeframe = small_data['YY/MM'].values
    output =  small_data['TSI(Chl-a)']

    training_set = training_set.drop('YY/MM', 1)
    test_set = test_set.drop('YY/MM', 1)
    training_set=training_set.astype('float')
    


    train_X = np.array(training_set[['NH3-N(㎎/L)', 'NO3-N(㎎/L)','DO (㎎/L)','BOD(㎎/L)',"I(pH)", 'Dissolved Total P(㎎/L)']])[:-1]
    train_y = np.array(training_set['TSI(Chl-a)'])[1:]


    test_X = np.array(test_set[['NH3-N(㎎/L)', 'NO3-N(㎎/L)','DO (㎎/L)','BOD(㎎/L)',"I(pH)", 'Dissolved Total P(㎎/L)']])[:-1]
    test_y = np.array(test_set['TSI(Chl-a)'])

    

    ano = 'Relative MSE: ' +str(mseValue)+'%', 
    lenValue.append([dictrict,len(small_data),mseValue])

    # fig, ax1 = plt.subplots(1,1,figsize=(20,10))
    # ax1.plot(timeframe[:len(real)],real, label='Actual')
    # ax1.plot(timeframe[:len(predictions)],predictions, label='Predicted')
    # ax1.annotate(ano, 
    #      xy=(0.75, 0.9),  xycoords='axes fraction',
    #     xytext=(0.75, 0.9), textcoords='axes fraction')
    # ax1.set_title("Dự đoán nổng độ tảo tại trạm "+dictrict,fontsize=13)
    # ax1.legend()
    # fig.autofmt_xdate()
    # ax1.set_ylim(bottom=0)
    # ax1.set_ylim(top=100)
    # # plt.show()
    # plt.savefig("sarima/"+ dictrict +'.png', dpi=100)

    final_predict.append(list(predictions[-2:]))
    true_label.append(list(real[-2:]))
    quit()


df = pd.DataFrame(list(zip(finalDictrict, finalMSE)), 
               columns =['Name', 'val']) 
df.to_csv("dtR.csv")

from sklearn.metrics import classification_report
print(classification_report(real_label, predict_label))
print(Average(finalMSE))
quit()
# totalCount = 0
# dictrictName = []

# dictrictMSE = []

# print(dictrictName)
print(final_predict)
print(true_label)

y_pred = []
print("final predict")
for i in range(0, len(lenValue)):
    if final_predict[i][0] > 60 or final_predict[i][1] >60:
        y_pred.append(1)
    else:
        y_pred.append(0)
        # grade = "Eutrophy"
        # if finalArr[i][0] > 70 or finalArr[i][1] >70:
        #   grade = "Hypereutrophy"
        # if finalArr[i][0] > 80 or finalArr[i][1] >80:
        #   grade = "Algae bloom"
        # print("tram "+ dictrictName[i]+ " co kha nang no hoa")
        # print("Grade:" + grade) 

y_true  = []
print("final predict")
for i in range(0, len(lenValue)):
    if true_label[i][0] > 60 or true_label[i][1] >60:
        y_true.append(1)
    else:
        y_true.append(0)
        # grade = "Eutrophy"
        # if finalArr2[i][0] > 70 or finalArr2[i][1] >70:
        #   grade = "Hypereutrophy"
        # if finalArr2[i][0] > 80 or finalArr2[i][1] >80:
        #   grade = "Algae bloom"
        # print("tram "+ dictrictName[i]+ " co kha nang no hoa")
        # print("Grade:" + grade) 

print(y_true)
print(len(y_true))
print(y_pred)
print(len(y_pred))


from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))

print(final_predict)
i= 0 
for row in lenValue:
    if final_predict[i][0] > 60 or final_predict[i][1] >60:
        grade = "Eutrophy"
        if final_predict[i][0] > 70 or final_predict[i][1] >70:
            grade = "Hypereutrophy"
        if final_predict[i][0] > 80 or final_predict[i][1] >80:
            grade = "Algae bloom"
        print("tram "+ row[2]+ " co kha nang no hoa")
        print("Grade:" + grade) 
    i = i +1
    # totalCount = totalCount + int(row[1])
    # print("Name: "+ row[0] + ", Count: " + str(row[1]) + ", Relative MSE:" + str(row[2])+"%")
    dictrictName.append(row[0])
    dictrictMSE.append(row[2])

# quit()
df = pd.DataFrame(list(zip(dictrictName, dictrictMSE)), 
               columns =['Name', 'val']) 
df.to_csv("res.csv")
# print(totalCount)

# print(dictrictLabel)
# print(dictrictValue)

# dictrictValue = np.array(dictrictValue).transpose().tolist()

# # print(len(dictrictValue))
# # print(len(dictrictLabel))


# df = pd.DataFrame(data =dictrictValue).transpose()
# df.columns = dictrictLabel
# # print(df)

# correlation = (df.corr())

# # correlation.to_csv("output.csv")


# correlation = correlation.values


# # print(correlation[0][0])

# for i in range(0,26):
#   for j in range(0,26):
#       if i==j:
#           continue
#       if correlation[i][j] > 0.7:
#           print(dictrictLabel[i] + "<--->" + dictrictLabel [j] +" ||| correlation: "+str(correlation[i][j]))