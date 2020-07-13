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


MasterDataframe = pd.read_excel('Data.xlsx')
MasterDataframe.rename(columns=MasterDataframe.iloc[0])
#Get all subset of the column
ColumnList = list(MasterDataframe)
print(ColumnList)


# =IF(BT2<30, "Clean water", IF(AND(BT2>=30, BT2<40), 
# "Hypolimia", IF(AND(BT2>=40,BT2<50), "Mesotrophic", 
# 	IF(AND(BT2>=50, BT2<70), "Eutrophic", 
# 		IF(BT2>70,"Hypertrophic")))))

print(MasterDataframe.head())


dataCol = ['Name (E)' ,'YY/MM','Dissolved Total N(㎎/L)', 'NH3-N(㎎/L)', 'NO3-N(㎎/L)', 'Dissolved Total P(㎎/L)','Conductivity(µS/㎝)','TSI(Chl-a)', 'Grade.3' ]

df = MasterDataframe[dataCol]

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
		print(col)
		# print("it here")
		df[col].fillna("Mesotrophic", inplace=True)

window_len = 5

myDf = df[dataCol[:-1]]
myDf = myDf.sort_values(by='YY/MM')

dictrictArr = (myDf['Name (E)'].unique())

training_input = np.empty((1,window_len,6))
training_output = np.empty((1,))
test_input = np.empty((1,window_len,6))
test_output = np.empty((1,))


# print(training_input)
# print(test_input)
# quit()
# norm_cols = ['NH3-N(㎎/L)', 'NO3-N(㎎/L)', 'Dissolved Total P(㎎/L)','TSI(Chl-a)']

near = ['Guui','Jamsil','Amsa']		
near = ['Gayang','Yangjaecheon Stream','Noryangjin']		
near = ['Jungnangcheon Stream 3','Jungnangcheon Stream 4','Jeongneungcheon Stream']		
near = ['','Jamsil','Bogwang']	

count = 0
dfArr = []

small_data = myDf[myDf['Name (E)']=='Jungnangcheon Stream 3']
small_data = small_data.drop('Name (E)', 1)

small_data2 = myDf[myDf['Name (E)']=='Jungnangcheon Stream 4']
small_data2 = small_data2.drop('Name (E)', 1)

small_data3 = myDf[myDf['Name (E)']=='Jeongneungcheon Stream']
small_data3 = small_data3.drop('Name (E)', 1)

print(small_data2)

myfinaldf = pd.merge(small_data,small_data2,left_on = 'YY/MM',right_on = 'YY/MM')
myfinaldf = pd.merge(myfinaldf,small_data3,left_on = 'YY/MM',right_on = 'YY/MM')



print(myfinaldf)


# print(list(myfinaldf))



norm_cols = ['NH3-N(㎎/L)', 'NO3-N(㎎/L)', 'Dissolved Total P(㎎/L)','TSI(Chl-a)',
'NH3-N(㎎/L)_x', 'NO3-N(㎎/L)_x', 'Dissolved Total P(㎎/L)_x','TSI(Chl-a)_x',
'NH3-N(㎎/L)_y', 'NO3-N(㎎/L)_y', 'Dissolved Total P(㎎/L)_y','TSI(Chl-a)_y',
]

count = count + 1
# print(small_data)
split_date = "2019/04"
myfinaldf = myfinaldf.sort_values(by='YY/MM')
training_set, test_set = myfinaldf[myfinaldf['YY/MM']<split_date], myfinaldf[myfinaldf['YY/MM']>=split_date]



timeframe = myfinaldf['YY/MM']
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

LSTM_training_outputs=training_set['TSI(Chl-a)_y'][window_len:].values

LSTM_test_inputs = []
for i in range(len(test_set)-window_len):
	temp_set = test_set[i:(i+window_len)].copy()
	for col in norm_cols:
		temp_set.loc[:, col] = temp_set[col]
	# print(temp_set)
	LSTM_test_inputs.append(temp_set)
	
LSTM_test_outputs = test_set['TSI(Chl-a)_y'][window_len:].values

LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)

LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)
# print(LSTM_test_inputs)

# quit()
my_model = build_LSTM_model(LSTM_training_inputs, output_size=1, neurons = 100)
my_model.fit(LSTM_training_inputs, LSTM_training_inputs, 
							epochs=50, batch_size=1, verbose=1, shuffle=True)


print(LSTM_test_inputs)
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
predict_based = test_set['TSI(Chl-a)_y'][window_len-1:].values

final_predict = []
final_predict.append(predict_based[1])

loop = 0
check = 0
# for i in range(1, len(predict_based)):
# 	if predict_based[i] == predict_based[i-1]:
# 		loop = loop +1
# 	if loop > 1:
# 		check = 1
# 		continue
# if check == 1:
# 	continue
for i in range(1, len(predict)):
	temp = (1 + (predict[i]-predict[i-1])/predict[i-1])*predict_based[i-1]
	final_predict.append(temp)
# print(final_predict)

fig, ax1 = plt.subplots(1,1,figsize=(20,10))
ax1.plot(timeframe[:len(test_set['TSI(Chl-a)_y'][window_len:].values,)],test_set['TSI(Chl-a)_y'][window_len:].values, label='Actual')
ax1.plot(timeframe[:len(final_predict)],final_predict, label='Predicted')
ax1.annotate('MAE: %.4f'%np.mean(np.abs((final_predict) - test_set['TSI(Chl-a)_y'][window_len:].values)), 
     xy=(0.75, 0.9),  xycoords='axes fraction',
    xytext=(0.75, 0.9), textcoords='axes fraction')

ax1.set_title("Dự đoán nổng độ tảo tại trạm ",fontsize=13)
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
plt.show()

# plt.savefig("LSTM/"+ dictrict +'.png', dpi=100)
# 	# quit()

