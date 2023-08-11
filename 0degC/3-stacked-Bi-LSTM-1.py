
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 11:11:39 2022

@author: yv5225
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout, LeakyReLU, ReLU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam, Adamax
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--array', type=int, required=True)
args = parser.parse_args()

number = np.arange(0,10,1)
number = number[args.array]

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
path = '/home/elham64/scratch/data/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/'


#Upsample dataset
namedeg = '0degC/'

charge_names_0 = [
    '589_Charge3',
    '589_Charge4',
    '589_Charge7',
    '589_Charge8',
    '590_Charge11',
    '590_Charge12',  
    '590_Charge13',
    '590_Charge14', 
    '590_Charge16',
    '590_Charge15',
     ]

discharge_names_0 = [    
    '589_Mixed1',
    '589_Mixed2',
    '590_Mixed4', 
    '590_Mixed5',
    '590_Mixed6',
    '590_Mixed7',
    '590_Mixed8',
    '590_PausCycl',
    '589_UDDS',
    '589_LA92',
    '589_US06',
      ]

for name in charge_names_0:
    
    cycle_ch = pd.read_csv(path + namedeg + name + '.csv', skiprows=29)
    cycle_ch.columns = ['Time Stamp','Step','Status','Prog Time','Step Time','Cycle',
                        'Cycle Level','Procedure','Voltage','Current','Temperature','Capacity','WhAccu','Cnt','Empty']
    if name == '590_Charge14':
        cycle_ch['Progtime'] = (pd.to_timedelta(cycle_ch['Prog Time'])- pd.Timedelta(hours=2)).astype(str).str.extract('days (.*?)\.') 
    else:  
        cycle_ch['Progtime'] = pd.to_timedelta(cycle_ch['Prog Time']).astype(str).str.extract('days (.*?)\.')
    
    
    cycle_ch['ProgTimeWithDate'] = pd.to_datetime(pd.to_datetime(cycle_ch['Time Stamp']).dt.date.astype(str) + ' ' + cycle_ch['Progtime'].astype(str))

    if name == '590_Charge12':
        cycle_ch['ProgTimeWithDate'] = cycle_ch['ProgTimeWithDate'].apply(lambda x:x.replace(day=1))
    
    
    cycle_ch.set_index(pd.DatetimeIndex(cycle_ch['ProgTimeWithDate']), inplace=True)
    cycle_ch = cycle_ch[['Voltage','Current','Temperature','Capacity']]
    
    #cycle_ch_drop = cycle_ch.drop_duplicates() 
    cycle_ch = cycle_ch.resample('S').mean().interpolate(method='linear')
    cycle_ch['Time'] = pd.to_timedelta(cycle_ch.index.strftime('%H:%M:%S')).astype('timedelta64[s]').astype(int)
    #empty_row = np.arange(0, 29, 1)
    #empty_cycle_ch = pd.DataFrame(np.nan, index=empty_row, columns=cycle_ch.columns)
    #cycle_up = pd.concat([empty_cycle_ch, cycle_ch])
    cycle_ch.to_csv(path + namedeg + 'upsampling/' + str(name) + '.csv')

for name_dch in discharge_names_0:
    cycle_dch = pd.read_csv(path + namedeg + name_dch + '.csv', skiprows=29)
    cycle_dch.columns = ['Time Stamp','Step','Status','Prog Time','Step Time','Cycle',
                    'Cycle Level','Procedure','Voltage','Current','Temperature','Capacity','WhAccu','Cnt','Empty']
    cycle_dch['Time'] = pd.to_timedelta(cycle_dch['Prog Time']).astype('timedelta64[s]').astype(int)
    cycle_dch = cycle_dch[['Voltage','Current','Temperature','Capacity','Time']] 
    cycle_dch.to_csv(path + namedeg + 'upsampling/' + str(name_dch) + '.csv')
            
            
#Load dataset
def get_data(names):
        cycles = []
        for name in names:
            cycle = pd.read_csv(path + name + '.csv', skiprows=30)
            cycle.columns = ['Time Stamp','Step','Status','Prog Time','Step Time','Cycle',
                            'Cycle Level','Procedure','Voltage','Current','Temperature','Capacity','WhAccu','Cnt','Empty']
            cycle['Time'] = pd.to_timedelta(cycle['Prog Time']).astype('timedelta64[s]').astype(int)
            
            #cycle['Time'] = pd.to_timedelta(cycle['Prog Time']).dt.total_seconds()            
         
            #cycle = cycle[(cycle["Status"] == "TABLE") | (cycle["Status"] == "DCH")]

            max_discharge = abs(min(cycle["Capacity"]))
            cycle["SoC Capacity"] = max_discharge + cycle["Capacity"]
            cycle["SoC Percentage"] = cycle["SoC Capacity"] / max(cycle["SoC Capacity"])
                        
            x = cycle[["Time", "Voltage", "Current", "Temperature"]].to_numpy()
            y = cycle[["Time","SoC Percentage"]].to_numpy()

            
            cycles.append((x, y))

        return cycles
    

def timeorder(cycles_order):
    x_length = len(cycles_order[0][0][0])
    y_length = len(cycles_order[0][1][0])
    x = np.zeros((0, x_length), float)
    y = np.zeros((0, y_length), float)
    prev_cycle_x = np.zeros((100, x_length), float)
    prev_cycle_y = np.zeros((100, y_length), float)
    print(prev_cycle_x[:,0][-1])
    print(prev_cycle_x[:,0][-1])
    for cycle_order in cycles_order:
        next_cycle_x = np.array(cycle_order[0])
        next_cycle_y = np.array(cycle_order[1])
        next_cycle_x[:,0] = next_cycle_x[:,0] - (next_cycle_x[:,0][0] - prev_cycle_x[:,0][-1])
        next_cycle_y[:,0] = next_cycle_y[:,0] - (next_cycle_y[:,0][0] - prev_cycle_y[:,0][-1])
        prev_cycle_x = next_cycle_x
        prev_cycle_y = next_cycle_y
        #rint(prev_cycle_x[:,0][0])
        #rint(prev_cycle_x[:,0][-1])
        x = np.concatenate((x, next_cycle_x))
        y = np.concatenate((y, next_cycle_y))
    return x, y

def average(xtraingroup, steps):
    x_V = np.zeros((0), float)  
    x_I = np.zeros((0), float)  
    for i in range(0, len(xtraingroup) - steps, 1):
        next_V = np.mean(xtraingroup[i:i + steps,0]).reshape(-1)
        next_I = np.mean(xtraingroup[i:i + steps,1]).reshape(-1)
        x_V = np.concatenate((x_V, next_V))
        x_I = np.concatenate((x_I, next_I))
    return x_V, x_I


def get_data_up(names):
        cycles = []
        for name in names:
            cycle = pd.read_csv(path + namedeg + 'upsampling/' + str(name) + '.csv')
            max_discharge = abs(min(cycle["Capacity"]))
            cycle["SoC Capacity"] = max_discharge + cycle["Capacity"]
            cycle["SoC Percentage"] = cycle["SoC Capacity"] / max(cycle["SoC Capacity"])
            
            
            
            x = cycle[["Time", "Voltage", "Current", "Temperature"]].to_numpy()
            y = cycle[["Time","SoC Percentage"]].to_numpy()

            
            cycles.append((x, y))

        return cycles
        
        
#Train and test dataset
train_names_0 = [
    '589_Mixed2',
    '589_Charge8',
    '590_Mixed4',
    '590_Charge11',
    '590_Mixed5',
    '590_Charge12',
    '590_Mixed6', 
    '590_Charge13',
    '590_Mixed7',
    '590_Charge14',
    '590_Mixed8',
    '590_Charge15',
    #'590_PausCycl',
    ]

test_names_0 = [
    '589_UDDS',
    '589_Charge3',
    '589_LA92', 
    '589_Charge4',
    '589_US06', 
    ]

cycles_train = get_data_up(train_names_0)
x_train_order, y_train_order = timeorder(cycles_train)
cycles_test = get_data_up(test_names_0)
x_test_order, y_test_order = timeorder(cycles_test)


df_xtrain = pd.DataFrame(x_train_order)
df_ytrain = pd.DataFrame(y_train_order)
df_ytrain.columns =['t', 'SOC']
df_xtrain.columns =['t', 'V', 'I', 'T']

df_xtest = pd.DataFrame(x_test_order)
df_ytest = pd.DataFrame(y_test_order)
df_ytest.columns =['t', 'SOC']
df_xtest.columns =['t', 'V', 'I', 'T']


cols_to_norm = ['V','I','T']
cols_to_norm_new = ['V_norm','I_norm','T_norm']
df_xtrain[cols_to_norm_new] = df_xtrain[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df_xtest[cols_to_norm_new] = df_xtest[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


df_xtrain_reset = df_xtrain.groupby(['t']).mean().reset_index()
df_xtrain_group = df_xtrain.groupby(['t']).mean()
df_ytrain_reset = df_ytrain.groupby(['t']).mean().reset_index()
df_ytrain_group = df_ytrain.groupby(['t']).mean()


df_xtest_reset = df_xtest.groupby(['t']).mean().reset_index()
df_xtest_group = df_xtest.groupby(['t']).mean()

df_ytest_reset = df_ytest.groupby(['t']).mean().reset_index()
df_ytest_group = df_ytest.groupby(['t']).mean()

xtrain_group = np.array(df_xtrain_group[['V_norm', 'I_norm']])
xtest_group = np.array(df_xtest_group[['V_norm', 'I_norm']])

k=400

V_avg, I_avg = average(xtrain_group, k)
V_avg_test, I_avg_test = average(xtest_group, k)

df_V_avg = pd.DataFrame(V_avg)
df_I_avg = pd.DataFrame(I_avg)
df_V_avg.columns =['V_avg']
df_I_avg.columns =['I_avg']

df_V_avg_test = pd.DataFrame(V_avg_test)
df_I_avg_test = pd.DataFrame(I_avg_test)
df_V_avg_test.columns =['V_avg']
df_I_avg_test.columns =['I_avg']

df_tot =  pd.concat([df_xtrain_reset, df_V_avg, df_I_avg], axis=1)
df_tot_test =  pd.concat([df_xtest_reset, df_V_avg_test, df_I_avg_test], axis=1)


df_tot['V_avg'] =df_tot.V_avg.shift(periods=k)
df_tot['I_avg'] =df_tot.I_avg.shift(periods=k)
df_tot.loc[:k, 'V_avg'] = df_tot['V_avg'][k+1]
df_tot.loc[:k, 'I_avg'] = df_tot['I_avg'][k+1]

df_tot_test['V_avg'] =df_tot_test.V_avg.shift(periods=k)
df_tot_test['I_avg'] =df_tot_test.I_avg.shift(periods=k)
df_tot_test.loc[:k, 'V_avg'] = df_tot_test['V_avg'][k+1]
df_tot_test.loc[:k, 'I_avg'] = df_tot_test['I_avg'][k+1]


dftrain_x = df_tot[['V_norm', 'I_norm', 'T_norm', 'V_avg', 'I_avg']]
dftest_x = df_tot_test[['V_norm', 'I_norm', 'T_norm', 'V_avg', 'I_avg']]
dftrain_y = df_ytrain_reset['SOC']
dftest_y = df_ytest_reset['SOC']



train_X = dftrain_x.to_numpy()
test_X = dftest_x.to_numpy()
train_y = dftrain_y.to_numpy()
test_y = dftest_y.to_numpy()



train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

train_y = train_y.reshape((train_y.shape[0]))
test_y = test_y.reshape((test_y.shape[0]))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape) 


n_features = 5
n_seq = 1
n_steps = 1
train_X = train_X.reshape((train_X.shape[0], n_seq, n_features, n_steps))
test_X = test_X.reshape((test_X.shape[0], n_seq, n_features, n_steps))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


n_steps, n_features = 1, 5

train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], n_features))

model = Sequential()
model.add(Bidirectional(LSTM(70, activation='relu', kernel_initializer= tf.keras.initializers.he_normal, return_sequences=True), input_shape=(n_steps, n_features)))
model.add(Bidirectional(LSTM(30, activation='relu', kernel_initializer= tf.keras.initializers.he_normal, return_sequences=False)))

#model.add(Dense(50))
model.add(Dense(1))

model.add(LeakyReLU(alpha=10e-9))
#model.add(ReLU(max_value=1.0))


opt = Adamax(learning_rate=0.0001)
model.summary()
model.compile(loss='mse', optimizer=opt)



# fit network
t0 = time.time()
history = model.fit(train_X, train_y, epochs=515, batch_size=512, validation_data=(test_X, test_y), verbose=1)
print("Training time:", time.time()-t0)



# summarize history for loss
fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
fig.savefig('loss-BiLSTM70to160-DD-Adamax-mse-CR.png')



# make a prediction
yhat = model.predict(test_X)



# calculate RMSE
testScore = np.sqrt(mean_squared_error(test_y, yhat))
print('Test Score: %.5f RMSE' % (testScore))


fig = plt.figure()
aa=[x for x in range(test_y.shape[0])]
plt.plot(aa, test_y, marker='.', label="actual")
plt.plot(aa, yhat, 'r', label="prediction")
plt.ylabel('SOC', size=15)
plt.xlabel('Time(s)', size=15)
plt.legend(fontsize=15)
fig.savefig('SOC-BiLSTM70to160-DD-Adamax-mse-CR.png')



test_resid = [test_y[i]-yhat[i] for i in range(len(yhat))]

fig = plt.figure()
aa=[x for x in range(test_y.shape[0])]
plt.plot(aa, test_resid, marker='.', label="actual")
plt.ylabel('error', size=15)
plt.xlabel('Time(s)', size=15)
plt.legend(fontsize=15)
plt.annotate("rmse = {:.5f}".format(testScore), (1000, 0.04))
fig.savefig('error-BiLSTM70to160-DD-Adamax-mse-CR.png')





