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
import time

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
path = '/home/elham64/scratch/data/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/'


#Upsample dataset
namedeg = 'n10degC/'

charge_names_N10 = [
    '601_Charge7',
    '602_Charge10',
    '602_Charge11',
    '602_Charge12',
    '604_Charge10',  
    '604_Charge13',
    '604_Charge14',
    '604_Charge15',
    '596_Charge4',
    '596_Charge5',
     ]

discharge_names_N10 = [
    
    '601_Mixed1',
    '601_Mixed2',
    '602_Mixed4', 
    '602_Mixed5',
    '604_Mixed3',
    '604_Mixed6',
    '604_Mixed7',
    '604_Mixed8',
    '604_PausCycl',
    '596_UDDS',
    '596_LA92',
    '601_US06',
      ]

for name in charge_names_N10:
    
    cycle_ch = pd.read_csv(path + namedeg + name + '.csv', skiprows=29)
    cycle_ch.columns = ['Time Stamp','Step','Status','Prog Time','Step Time','Cycle',
                    'Cycle Level','Procedure','Voltage','Current','Temperature','Capacity','WhAccu','Cnt','Empty']
    
    if name == '604_Charge15':
        cycle_ch['Progtime'] = (pd.to_timedelta(cycle_ch['Prog Time'])- pd.Timedelta(hours=3)).astype(str).str.extract('days (.*?)\.') 
    else:  
        cycle_ch['Progtime'] = pd.to_timedelta(cycle_ch['Prog Time']).astype(str).str.extract('days (.*?)\.')
    
    
    cycle_ch['ProgTimeWithDate'] = pd.to_datetime(pd.to_datetime(cycle_ch['Time Stamp']).dt.date.astype(str) + ' ' + cycle_ch['Progtime'].astype(str))

    if name == '604_Charge16':
        cycle_ch['ProgTimeWithDate'] = cycle_ch['ProgTimeWithDate'].apply(lambda x:x.replace(day=18))
    
    
    cycle_ch.set_index(pd.DatetimeIndex(cycle_ch['ProgTimeWithDate']), inplace=True)
    cycle_ch = cycle_ch[['Voltage','Current','Temperature','Capacity']]
    
    #cycle_ch_drop = cycle_ch.drop_duplicates() 
    cycle_ch = cycle_ch.resample('S').mean().interpolate(method='linear')
    cycle_ch['Time'] = pd.to_timedelta(cycle_ch.index.strftime('%H:%M:%S')).astype('timedelta64[s]').astype(int)
    #empty_row = np.arange(0, 29, 1)
    #empty_cycle_ch = pd.DataFrame(np.nan, index=empty_row, columns=cycle_ch.columns)
    #cycle_up = pd.concat([empty_cycle_ch, cycle_ch])
    cycle_ch.to_csv(path + namedeg + 'upsampling/' + str(name) + '.csv')
    

for name_dch in discharge_names_N10:
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
            cycle = pd.read_csv(path + name + '.csv', skiprows=29)
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
train_names_N10 = [
    '601_Mixed1',
    '601_Charge7',
    '601_Mixed2',
    '602_Charge10',
    '602_Mixed4', 
    '602_Charge11',
    '602_Mixed5',
    '602_Charge12',
    '604_Mixed3',
    '602_Charge10',
    '604_Mixed6',
    '604_Charge13',
    '604_Mixed7',
    '604_Charge14',
    '604_Mixed8',
    '604_Charge15',
    #'604_PausCycl',
    #'604_Charge16',
    ]

test_names_N10 = [
    '596_UDDS',
    '596_Charge4',
    '596_LA92',
    '596_Charge5',
    '601_US06',
    ]


cycles_train = get_data_up(train_names_N10)
x_train_order, y_train_order = timeorder(cycles_train)
cycles_test = get_data_up(test_names_N10)
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

n_steps, n_features = 1, 5


def create_lstm(optimizations,latent_dimension,loss):
    unit1 = latent_dimension[0] 
    unit2 = latent_dimension[1]   
    model = Sequential()
    model.add(LSTM(unit1, activation='relu', return_sequences=True, kernel_initializer= tf.keras.initializers.he_normal, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(LSTM(unit2, activation='relu', kernel_initializer= tf.keras.initializers.he_normal))
    model.add(Dense(1))
    model.add(LeakyReLU(alpha=10e-9))
    #model.add(ReLU(max_value=1.0))
    print("Params: "+str(model.count_params()))
    model.compile(optimizer=optimizations, loss=loss)
    return model
        
def Optimize(opts,dim,loss,batch=512,epoch=515,verbose=0):
    model_generated = " batch: "+str(batch)+" | dim: "+str(dim)+" | loss: "+str(loss)+" | epoch: "+str(epoch)      
    model = None
    model = create_lstm(opts,dim,loss)
    t0 = time.time()
    history_temp = model.fit(train_X, train_y, epochs=epoch, batch_size=batch, validation_data=(test_X, test_y), verbose=verbose)
    print("Training time:", time.time()-t0)
    #print('Predict')
    y_predict = model.predict(test_X,verbose=0)
    print('compute RMSE,MAE')
    y_predict = y_predict.reshape(y_predict.shape[0]) 
    rmse_temp = np.sqrt(mean_squared_error(test_y, y_predict))
    temo_loss = " | loss: " + str(history_temp.history['loss'][epoch-1]) + " | val_loss: " + str(history_temp.history["val_loss"][epoch-1])
    MAE = mean_absolute_error(test_y, y_predict)
    temp_result = "rmse: "+str(rmse_temp)+"MAE:" + str(MAE) +" " + temo_loss +" " + model_generated
    print(opts)
    print(dim)
    print(loss)
    print(temp_result) 
    fig = plt.figure()
    plt.plot(history_temp.history['loss'])
    plt.plot(history_temp.history['val_loss'])
    plt.title('model loss')
    plt.yscale('log')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.annotate("rmse = {:.5f} , mae = {:.5f}, training time = {:.3f}".format(rmse_temp, MAE, (time.time()-t0)/ 60.0), (100, 0.01))
    plt.savefig('stacked-LSTM-2layer-dim: '+str(dim)+' loss: '+str(loss)+' opt: '+str(opts)+'-2.png') 
    plt.close()
    return temp_result    

latent_dimension=[[25,25],[40,20],[50,25],[50,50],[70,50],[70,70],[90,70],[100,50],[110,90]]
loss_function=['mse']
optimizations=[Adam(learning_rate=0.0001)]


result = []
for opts in optimizations:
    for dim in latent_dimension:
        for loss in loss_function:
            result.append(Optimize(opts,dim,loss))



        
        
        
        
        
        
        
        
        







