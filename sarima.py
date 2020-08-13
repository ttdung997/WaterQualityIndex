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

from matplotlib.dates import (YEARLY, DateFormatter,
                              rrulewrapper, RRuleLocator, drange)



# dataCol = ['Name (E)' ,'YY/MM','Dissolved Total N(㎎/L)', 'NH3-N(㎎/L)', 'NO3-N(㎎/L)', 'Dissolved Total P(㎎/L)','Conductivity(µS/㎝)','TSI(Chl-a)', 'Grade.3' ]

dataCol = ['Name (E)' ,'YY/MM','NH3-N(㎎/L)', 'NO3-N(㎎/L)', 'PO4-P(㎎/L)',
 'T-N(㎎/L)','T-P(㎎/L)', 'Dissolved Total N(㎎/L)','Dissolved Total P(㎎/L)',
  'Hydrogen ion conc.','DO (㎎/L)', 'TSI(Chl-a)']

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
# 	if x not  in trash:
# 		dictrictArr.append(x)

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

count = 0
lenValue = []
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
	# print(small_data)
	split_date = "2020/01"
	training_set, test_set = small_data[small_data['YY/MM']<"2019/09"], small_data[small_data['YY/MM']>="2019/09"]


	timeframe = small_data['YY/MM'].values
	output =  small_data['TSI(Chl-a)']

	training_set = training_set.drop('YY/MM', 1)
	test_set = test_set.drop('YY/MM', 1)
	training_set=training_set.astype('float')
	


	train_X = np.array(training_set[['NH3-N(㎎/L)', 'NO3-N(㎎/L)', 'Dissolved Total P(㎎/L)']])[:-1]
	train_y = np.array(training_set['TSI(Chl-a)'])[1:]


	test_X = np.array(test_set[['NH3-N(㎎/L)', 'NO3-N(㎎/L)', 'Dissolved Total P(㎎/L)']])[:-1]
	test_y = np.array(test_set['TSI(Chl-a)'])

	from statsmodels.tsa.statespace.sarimax import SARIMAX
	model= SARIMAX(train_y, 
	 exog= train_X, 
	 order=(3,1,1),
	 enforce_invertibility=False, enforce_stationarity=False)
	results = model.fit()


	predictions= results.predict(start = len(train_y)  -13 , end= len(train_y) +2, exog= test_X[:3])

	

	real = output[len(train_y)-13:len(train_y) +2]

	dates = [i for i in range(0,len(predictions))]

	mseValue =(np.mean(np.abs(predictions[:len(real)] - real))/np.min(predictions))

	if mseValue == 0.0:
		continue

	ano = 'Relative MSE: ' +str(mseValue)+'%', 
	lenValue.append([dictrict,len(small_data),mseValue])

	fig, ax1 = plt.subplots(1,1,figsize=(20,10))
	ax1.plot(timeframe[:len(real)],real, label='Actual')
	ax1.plot(timeframe[:len(predictions)],predictions, label='Predicted')
	ax1.annotate(ano, 
         xy=(0.75, 0.9),  xycoords='axes fraction',
        xytext=(0.75, 0.9), textcoords='axes fraction')
	ax1.set_title("Dự đoán nổng độ tảo tại trạm "+dictrict,fontsize=13)
	ax1.legend()
	fig.autofmt_xdate()
	ax1.set_ylim(bottom=0)
	ax1.set_ylim(top=100)
	# plt.show()
	plt.savefig("sarima/"+ dictrict +'.png', dpi=100)

	final_predict.append(list(predictions[-2:]))
	true_label.append(list(real[-2:]))
	# quit()


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
		# 	grade = "Hypereutrophy"
		# if finalArr[i][0] > 80 or finalArr[i][1] >80:
		# 	grade = "Algae bloom"
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
		# 	grade = "Hypereutrophy"
		# if finalArr2[i][0] > 80 or finalArr2[i][1] >80:
		# 	grade = "Algae bloom"
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
df.to_csv("sarima.csv")
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
# 	for j in range(0,26):
# 		if i==j:
# 			continue
# 		if correlation[i][j] > 0.7:
# 			print(dictrictLabel[i] + "<--->" + dictrictLabel [j] +" ||| correlation: "+str(correlation[i][j]))