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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.layers import Dropout


from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import SimpleRNN

from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import sequence
from scipy import interp
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from sklearn.linear_model import LinearRegression
from matplotlib.dates import (YEARLY, DateFormatter,
	rrulewrapper, RRuleLocator, drange)
#load excel dataframe

TRAINING_FLAG = 0




def build_LSTM_model(inputs, output_size, neurons, activ_func="linear",
	dropout=0.25, loss="mae", optimizer="adam"):
	model = Sequential()

	model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
	model.add(Dropout(dropout))
	model.add(Dense(units=output_size))
	model.add(Activation(activ_func))

	model.compile(loss=loss, optimizer=optimizer)
	return model

def build_RNN_model(inputs, output_size, neurons, activ_func="linear",
	dropout=0.25, loss="mae", optimizer="adam"):
	model = Sequential()

	model.add(SimpleRNN(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
	model.add(Dropout(dropout))
	model.add(Dense(units=output_size))
	model.add(Activation(activ_func))
	model.compile(loss=loss, optimizer=optimizer)
	return model


def build_GRU_model(inputs, output_size, neurons, activ_func="linear",
	dropout=0.25, loss="mae", optimizer="adam"):
	model = Sequential()

	model.add(GRU(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
	model.add(Dropout(dropout))
	model.add(Dense(units=output_size))
	model.add(Activation(activ_func))

	model.compile(loss=loss, optimizer=optimizer)
	return model



# =9.81*LN(AH13)+30.6
# =IF(BT2<30, "Clean water", IF(AND(BT2>=30, BT2<40), 
# "Hypolimia", IF(AND(BT2>=40,BT2<50), "Mesotrophic", 
# 	IF(AND(BT2>=50, BT2<70), "Eutrophic", 
# 		IF(BT2>70,"Hypertrophic")))))

# dataCol = ['Name (E)' ,'YY/MM','Dissolved Total N(㎎/L)', 'NH3-N(㎎/L)', 'NO3-N(㎎/L)', 'Dissolved Total P(㎎/L)','Conductivity(µS/㎝)','TSI(Chl-a)', 'Grade.3' ]

# NH3-N(㎎/L)
# NO3-N(㎎/L)
# PO4-P(㎎/L)
# T-N(㎎/L)
# T-P(㎎/L)
# Dissolved Total N(㎎/L)
# Dissolved Total P(㎎/L)
# Hydrogen ion conc.
# DO (㎎/L)

dataCol = ['Name (E)' ,'YY/MM','NH3-N(㎎/L)', 'NO3-N(㎎/L)', 'PO4-P(㎎/L)',
 'T-N(㎎/L)','T-P(㎎/L)', 'Dissolved Total N(㎎/L)','Dissolved Total P(㎎/L)',
  'Hydrogen ion conc.','DO (㎎/L)', 'TSI(Chl-a)']


# dataCol = ['Name (E)' ,'YY/MM','Dissolved Total N(㎎/L)', 'NH3-N(㎎/L)', 'NO3-N(㎎/L)', 'Dissolved Total P(㎎/L)','Conductivity(µS/㎝)','TSI(Chl-a)']


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


correlation = (df.corr())

# correlation.to_csv("output.csv")

# print(correlation)
# correlation = correlation.values

### Calculate correlation

# # print(correlation[0][0])
# print(len(dataCol))
# for i in range(0,len(correlation)-1):
# 	for j in range(0,len(correlation)-1):
# 		if i==j:
# 			continue
# 		if correlation[i][j] > 0:
# 			print(dataCol[i] + "<--->" + dataCol [j] +" ||| correlation: "+str(correlation[i][j]))

# quit()


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
		continue
		# print(col)
		# print("it here")
		# df[col].fillna("Mesotrophic", inplace=True)

window_len = 5

myDf = df[dataCol]
myDf = myDf.sort_values(by='YY/MM')

dictrictArr = (myDf['Name (E)'].unique())

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
count = 0

dictrictMSE = []
dictrictName =  []

for dictrict in dictrictArr:
	try:
		print(dictrict)
		count = count + 1
		small_data = df[df['Name (E)']==dictrict]
		small_data = small_data.drop('Name (E)', 1)
		timeframe = small_data['YY/MM'].values
		# print(list(small_data))
		split_date = "2018/09"
		training_set, test_set = small_data[small_data['YY/MM']<split_date], small_data[small_data['YY/MM']>=split_date]

		training_set = training_set.drop('YY/MM', 1)
		test_set = test_set.drop('YY/MM', 1)
		training_set=training_set.astype('float')

		test_set=test_set.astype('float')

		# print(training_set)
		# print(test_set)

		LSTM_training_inputs = []
		for i in range(len(training_set)-window_len):
			temp_set = training_set[i:(i+window_len)].copy()
			for col in norm_cols:
				temp_set.loc[:, col] = temp_set[col]
				# print(temp_set)
				# LSTM_training_inputs = []
			LSTM_training_inputs.append(temp_set)


		LSTM_test_inputs = []
		for i in range(len(test_set)-window_len):
			temp_set = test_set[i:(i+window_len)].copy()
			for col in norm_cols:
				temp_set.loc[:, col] = temp_set[col]
		# print(temp_set)
			LSTM_test_inputs.append(temp_set)
		
		LSTM_test_outputs = test_set['TSI(Chl-a)'][window_len:].values

		LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
		LSTM_training_inputs = np.array(LSTM_training_inputs)

		LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
		LSTM_test_inputs = np.array(LSTM_test_inputs)

		# print(LSTM_last_input.shape)
		# LSTM_last_input.to_csv("lastdata.csv")
		# LSTM_last_input = LSTM_test_inputs[-1]
		# LSTM_last_input.shape = (1,10,4)

		np.random.seed(202)
		LSTM_training_outputs = training_set['TSI(Chl-a)'][window_len:].values

		# print("213123")
		# print(LSTM_training_inputs.shape)
		# print(LSTM_training_outputs.shape)
		# print(training_input)
		# print(LSTM_training_inputs)
		if count > 1:
			training_input  = np.concatenate((training_input, LSTM_training_inputs), axis=0) 
			training_output = np.concatenate((training_output, LSTM_training_outputs), axis=0) 
			test_input = np.concatenate((test_input, LSTM_test_inputs), axis=0) 
			test_output = np.concatenate((test_output, LSTM_test_outputs), axis=0) 
		else:
			training_input = LSTM_training_inputs 
			training_output = LSTM_training_outputs 
			test_input = LSTM_test_inputs 
			test_output = LSTM_test_outputs 
		# my_model = ""
		# my_model = build_LSTM_model(LSTM_training_inputs, output_size=1, neurons = 35)
		# my_model.fit(LSTM_training_inputs, LSTM_training_outputs, 
		# 	epochs=30, batch_size=1, verbose=1, shuffle=True)
		# predict =  my_model.predict(LSTM_test_inputs)

		# print(len(predict))

		# dates = [1,2,3,4,5,6,7,8]
		# # fig, ax1 = plt.subplots(1,1)
		# # print(LSTM_test_inputs)
		# # print("***************************************")
		# # print(predict)
		# # print(test_set['TSI(Chl-a)'][window_len-1:].values)
		# # print("____________________________________________________")
		# # print(LSTM_test_outputs)
		# predict_based = test_set['TSI(Chl-a)'][window_len-1:].values

		# final_predict = []
		# final_predict.append(predict_based[1])

		# loop = 0
		# check = 0
		# for i in range(1, len(predict_based)):
		# 	if predict_based[i] == predict_based[i-1]:
		# 		loop = loop +1
		# 		if loop > 1:
		# 			check = 1
		# 			continue
		# if check == 1:
		# 	continue
		# for i in range(1, len(predict)):
		# 	temp = (1 + (predict[i]-predict[i-1])/predict[i-1])*predict_based[i-1]
		# 	final_predict.append(temp)
		# print(final_predict)
		# mseValue = np.mean(np.abs((final_predict) - test_set['TSI(Chl-a)'][window_len:].values))/max(final_predict)
		
		# fig, ax1 = plt.subplots(1,1,figsize=(20,10))
		# ax1.plot(timeframe[:len(test_set['TSI(Chl-a)'][window_len:].values,)],test_set['TSI(Chl-a)'][window_len:].values, label='Actual')
		# ax1.plot(timeframe[:len(final_predict)],final_predict, label='Predicted')
		# ax1.annotate('MAE: %.4f'%mseValue, 
		# 	xy=(0.75, 0.9),  xycoords='axes fraction',
		# 	xytext=(0.75, 0.9), textcoords='axes fraction')

		# ax1.set_title("Dự đoán nổng độ tảo tại trạm "+dictrict,fontsize=13)
		# ax1.legend()
		# fig.autofmt_xdate()
		# ax1.set_ylim(bottom=0)
		# ax1.set_ylim(top=100)
		# # ax1.set_ylabel('gía cổ phiếu (VND)',fontsize=12)
		# # ax1.xaxis.set_major_locator(loc)
		# # ax1.xaxis.set_major_formatter(formatter)
		# # ax1.xaxis.set_tick_params(rotation=10, labelsize=10)
		# # ax1.set_ylim(bottom=0)
		# # ax1.set_ylim(top=100)
		# # plt.show()

		# dictrictName.append(dictrict)
		# dictrictMSE.append(mseValue)

		# plt.savefig("LSTM-tiny/"+ dictrict +'.png', dpi=100)

	except:
		continue

# for i in range(0, len(dictrictName)):
# 	print(dictrictName[i] + "||" + str(dictrictMSE[i]) + "|||")

# df = pd.DataFrame(list(zip(dictrictName, dictrictMSE)), 
#                columns =['Name', 'val']) 
# df.to_csv("tiny.csv")

# quit()
# print(len(training_input))
# print(len(training_output))
# quit()
# my_model = build_LSTM_model(training_input, output_size=1, neurons = 100)
# my_model.fit(training_input, training_output, 
# 	epochs=50, batch_size=1, verbose=1, shuffle=True)

# model_json =  my_model.to_json()
# model_output = "model/rnn_model.json"
# weight_output = "model/rnn_model.h5"
# with open(model_output, "w") as json_file:
#         json_file.write(model_json)
#         # serialize weights to HDF5
#         my_model.save_weights(weight_output)

model_output = 'model/rnn_model.json'
weight_output = 'model/rnn_model.h5'
json_file = open(model_output, 'r')
loaded_model_json = json_file.read()
json_file.close()
my_model = model_from_json(loaded_model_json)
my_model.load_weights(weight_output)

# print("_________________________")

# print(len(dictrictArr))
# print(len(myDf))
finalArr = []
for dictrict in dictrictArr:
	try:
		count = count + 1
		small_data = df[df['Name (E)']==dictrict]
		# print(len(small_data))
		small_data = small_data.drop('Name (E)', 1)
		timeframe = small_data['YY/MM'].values
		# print(small_data)
		split_date = "2016/04"
		training_set, test_set = small_data[small_data['YY/MM']<split_date], small_data[small_data['YY/MM']>=split_date]
		print(len(test_set))
		# continue
		training_set = training_set.drop('YY/MM', 1)
		test_set = test_set.drop('YY/MM', 1)
		training_set=training_set.astype('float')

		test_set=test_set.astype('float')

		
		LSTM_training_inputs = []
		for i in range(len(training_set)-window_len):
			temp_set = training_set[i:(i+window_len)].copy()
			for col in norm_cols:
				temp_set.loc[:, col] = temp_set[col]
				#print(temp_set)
			LSTM_training_inputs.append(temp_set)


		LSTM_test_inputs = []
		for i in range(len(test_set)-window_len):
			temp_set = test_set[i:(i+window_len)].copy()
			for col in norm_cols:
				temp_set.loc[:, col] = temp_set[col]
		# print(temp_set)
			LSTM_test_inputs.append(temp_set)
		
		LSTM_test_outputs = test_set['TSI(Chl-a)'][window_len:].values

		LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
		LSTM_training_inputs = np.array(LSTM_training_inputs)

		LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
		LSTM_test_inputs = np.array(LSTM_test_inputs)
		# print(LSTM_test_inputs)


		predict =  my_model.predict(LSTM_test_inputs)

		# print(len(predict))

		dates = [1,2,3,4,5,6,7,8]
		# fig, ax1 = plt.subplots(1,1)
		# print(LSTM_test_inputs)
		# print("***************************************")
		# print(predict)
		# print(test_set['TSI(Chl-a)'][window_len-1:].values)
		# print("____________________________________________________")
		# print(LSTM_test_outputs)
		predict_based = test_set['TSI(Chl-a)'][window_len-1:].values

		print(predict)

		print(predict_based)

		final_predict = []
		final_predict.append(predict_based[1])

		loop = 0
		check = 0
		# for i in range(1, len(predict_based)):
		# 	if predict_based[i] == predict_based[i-1]:
		# 		loop = loop +1
		# 		if loop > 1:
		# 			check = 1
		# 			continue
		# 	if check == 1:
		# 		continue
		for i in range(1, len(predict)):
			temp = (1 + (predict[i]-predict[i-1])/predict[i-1])*predict_based[i-1]
			final_predict.append(temp)
		print(final_predict)
		

		mseValue = np.mean(np.abs((final_predict) - test_set['TSI(Chl-a)'][window_len:].values))/max(final_predict)
		
		fig, ax1 = plt.subplots(1,1,figsize=(20,10))
		ax1.plot(timeframe[:len(test_set['TSI(Chl-a)'][window_len:].values,)],test_set['TSI(Chl-a)'][window_len:].values, label='Actual')
		ax1.plot(timeframe[:len(final_predict)],final_predict, label='Predicted')
		ax1.annotate('MAE: %.4f'%mseValue, 
			xy=(0.75, 0.9),  xycoords='axes fraction',
			xytext=(0.75, 0.9), textcoords='axes fraction')

		ax1.set_title("Dự đoán nổng độ tảo tại trạm "+dictrict,fontsize=13)
		ax1.legend()
		fig.autofmt_xdate()
		ax1.set_ylim(bottom=0)
		ax1.set_ylim(top=100)
		# ax1.set_ylabel('gía cổ phiếu (VND)',fontsize=12)
		# ax1.xaxis.set_major_locator(loc)
		# ax1.xaxis.set_major_formatter(formatter)
		# ax1.xaxis.set_tick_params(rotation=10, labelsize=10)
		# ax1.set_ylim(bottom=0)
		# ax1.set_ylim(top=100)
		# plt.show()
		finalArr.append(final_predict[-2:])
		dictrictName.append(dictrict)
		dictrictMSE.append(mseValue)

		plt.savefig("LSTM/"+ dictrict +'.png', dpi=100)
	# 	# quit()
	except:
		continue


for i in range(0, len(dictrictName)):
	print(dictrictName[i] + "||" + str(dictrictMSE[i]) + "|||")

df = pd.DataFrame(list(zip(dictrictName, dictrictMSE)), 
               columns =['Name', 'val']) 
df.to_csv("tiny.csv")

# 30-40	0,95-2,6	Hypolimia: 
# 40-50	2,6-7,3	Alpha- Mesotrophy
# 50-60	7,3-20	Beta- Mesotrophy
# 60-70	20-56	Eutrophy
# 70-80	56-155	Hypereutrophy
# >80	>155	Algae bloom

print("final predict")
for i in range(0, len(dictrictName)):
	if finalArr[i][0] > 60 or finalArr[i][1] >60:
		grade = "Eutrophy"
		if finalArr[i][0] > 70 or finalArr[i][1] >70:
			grade = "Hypereutrophy"
		if finalArr[i][0] > 80 or finalArr[i][1] >80:
			grade = "Algae bloom"
		print("tram "+ dictrictName[i]+ " co kha nang no hoa")
		print("Grade:" + grade) 
