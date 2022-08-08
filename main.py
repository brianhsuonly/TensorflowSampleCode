import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib
import pandas as pd
import os
import keras as K

from keras import datasets, layers, optimizers, Sequential, metrics, activations
from keras.layers import Bidirectional, TimeDistributed, RepeatVector
from keras.utils import plot_model
from datetime import datetime, timedelta
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def root_mean_squared_error(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.losses.MSE(y_true, y_pred))


def normalize(df):
    norm = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return norm


def renormalize(array, min, max):
    for i in range(0, array.size):
        array[i] = array[i] * (max - min) + min
    return array

def train_windows(df, ref_day=5, predict_day=1):
    X_train, Y_train = [], []
    for i in range(df.shape[0] - predict_day - ref_day + 1):
        X_train.append(np.array(df.iloc[i:i + ref_day, :-1]))
        Y_train.append(np.array(df.iloc[i + ref_day:i + ref_day + predict_day]["y"]))
    return np.array(X_train), np.array(Y_train)


def shuffle(X, Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]


class stock_lstm(K.Model):
    def __init__(self, units):
        super(stock_lstm, self).__init__()
        self.rnn = Sequential(
            [layers.SimpleRNN(units, input_shape=(refDay, 309), dropout=0.2, return_sequences=True, unroll=True),
             layers.SimpleRNN(units, dropout=0.2, unroll=True)])
        self.fc = layers.Dense(64)
        self.out = layers.Dense(1)

    def call(self, inputs, training=None):
        x = self.rnn(inputs)
        x = self.fc(x)
        out = self.out(x)
        return out


def build_model(maxFeature):

    model = Sequential()
    model.add(layers.Conv1D(filters=Layers[1], kernel_size=1, input_shape=(refDay, maxFeature), padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=1))
    model.add(layers.Dropout(0.2))

    model.add(layers.LSTM(Layers[2]))
    model.add(layers.Dropout(0.2))

    model.add(RepeatVector(preDay))
    model.add(TimeDistributed(layers.Dense(1)))
    model.add(layers.Activation("linear"))

    return model


def add_feature(entireDF, feature=None, FilePath=None, featureName=''):
    if FilePath != None:
        feature = pd.read_csv(FilePath, encoding='big5hkscs')
        feature['Date'] = pd.to_datetime(feature['Date'])
        feature.set_index(keys=['Date'], inplace=True)
        for i in range(feature.shape[1]):
            feature.rename(columns={feature.columns[i]: feature.columns[i] + '_' + str(featureName)}, inplace=True)
    # print(feature)
    # entireDF=pd.concat([entireDF,feature],axis=1,sort=False)
    entireDF = entireDF.merge(feature, left_index=True, right_index=True, how='outer')
    return entireDF


def extendStockData(stock):
    extendStock = pd.DataFrame(columns=["Date"])
    extendStock.set_index(keys=["Date"], inplace=True)

    for stockID in stock.index.get_level_values('證券代號').unique():
        nowStock = stock.loc[stockID]
        for i in range(nowStock.shape[1]):
            # print(nowStock.columns[i])
            nowStock.rename(columns={nowStock.columns[i]: nowStock.columns[i] + '_' + str(int(stockID))}, inplace=True)
        extendStock = pd.concat([extendStock, nowStock], axis=1, sort=False)

    return extendStock


def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


if __name__ == '__main__':

    solve_cudnn_error()
    startTime = datetime(2014, 2, 18)
    endTime = datetime(2020, 11, 30)
    refDay = 90
    preDay = 5
    testDay = 615
    batchsz = 64
    units = 256
    epochs = 300
    Layers = [1, 128, 256, preDay]
    loss = ['mean_absolute_error', 'mean_squared_error']
    target = '收盤價_1301'

    root = "DBpath"

    financialReportPath = os.path.join(root, 'financialReport')
    statisticsPath = os.path.join(root, 'statistics')
    tempPath = os.path.join(root, 'temp')

    filepath = "model\\ref=" + str(refDay) + ' , pre=' + str(preDay) + ' , batch=' + str(batchsz) + ' , epochs=' + str(
        epochs) + ' , layers=' + str(Layers) + '\\'
    if not os.path.isdir(root + filepath):
        os.mkdir(root + filepath)

    allData = pd.DataFrame(columns=["Date"])
    allData.set_index(keys=["Date"], inplace=True)

    stock = pd.read_csv(root + 'statistics\\每日股票交易-complete.csv', encoding='big5hkscs')
    stock['Date'] = pd.to_datetime(stock['Date'])
    stock.drop(columns=['證券名稱'], inplace=True)
    stock.set_index(keys=["證券代號", "Date"], inplace=True)

    extendStock = extendStockData(stock)
    allData = add_feature(allData, extendStock)

    allData = add_feature(allData, FilePath=root + 'oil\\布蘭特原油現貨價.csv', featureName='布蘭特原油現貨價')
    allData = add_feature(allData, FilePath=root + 'oil\\布蘭特原油期貨.csv', featureName='布蘭特原油期貨')
    allData = add_feature(allData, FilePath=root + 'oil\\西德州原油現貨價.csv', featureName='西德州原油現貨價')
    allData = add_feature(allData, FilePath=root + 'oil\\紐約輕原油期貨.csv', featureName='紐約輕原油期貨')
    allData = add_feature(allData, FilePath=root + 'oil\\IMF杜拜原油.csv', featureName='IMF杜拜原油')
    allData = add_feature(allData, FilePath=root + 'oil\\紐約杜拜原油期貨.csv', featureName='紐約杜拜原油期貨')
    allData = add_feature(allData, FilePath=root + 'oil\\美國API當週原油庫存變化.csv', featureName='美國API當週原油庫存變化')
    allData = add_feature(allData, FilePath=root + 'oil\\美國當周石油活躍鑽井數.csv', featureName='美國當周石油活躍鑽井數')
    allData = add_feature(allData, FilePath=root + 'oil\\美國EIA當週原油庫存變化.csv', featureName='美國EIA當週原油庫存變化')
    allData = add_feature(allData, FilePath=root + 'oil\\美國TRCC CRB天然氣指數.csv', featureName='美國TRCC CRB天然氣指數')
    allData = add_feature(allData, FilePath=root + 'oil\\美國當周天然氣活躍鑽井數.csv', featureName='美國當周天然氣活躍鑽井數')
    allData = add_feature(allData, FilePath=root + 'oil\\美國EIA當週天然氣庫存量.csv', featureName='美國EIA當週天然氣庫存量')
    allData = add_feature(allData, FilePath=root + 'oil\\美國天然氣ETF.csv', featureName='美國天然氣ETF')
    allData = add_feature(allData, FilePath=root + 'oil\\美國天然氣期貨.csv', featureName='美國天然氣期貨')

    allData = allData.loc[startTime:endTime]
    allData['year'] = allData.index.get_level_values('Date').year
    allData['month'] = allData.index.get_level_values('Date').month
    allData['day'] = allData.index.get_level_values('Date').day

    allData.to_csv(os.path.join(statisticsPath, 'LSTMDataOriginal.csv'), encoding='big5hkscs')
    allData['y'] = extendStock[target].shift(-1)
    priceMin = np.min(extendStock[target])
    priceMax = np.max(extendStock[target])

    allData.fillna(method='pad', inplace=True)
    allData.fillna(0.0, inplace=True)
    allData.iloc[:, :] = normalize(allData.iloc[:, :])
    allData.dropna(axis=0, how='all', inplace=True)
    allData.dropna(axis=1, how='all', inplace=True)

    numFeature = allData.shape[1] - 1
    allData.to_csv(os.path.join(statisticsPath, 'LSTMData.csv'), encoding='big5hkscs')

    '''gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)'''

    train = allData[:(-testDay)]
    test = allData[(-testDay):]

    X_train, Y_train = train_windows(train, refDay, preDay)
    X_train, Y_train = shuffle(X_train, Y_train)
    X_test, Y_test = train_windows(test, refDay, preDay)

    data = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    data = data.batch(batchsz, drop_remainder=True)

    data_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    data_test = data_test.batch(batchsz, drop_remainder=True)

    data_iter = iter(data)
    samples = next(data_iter)
    print(samples[0].shape, samples[1].shape)

    # rnn_model = stock_lstm(Layers[1])
    rnn_model = build_model(numFeature)
    rnn_model.compile(optimizer=K.optimizers.Adam(), loss=loss[0], metrics=[loss[1]])
    '''rnn_model.compile(optimizer  =K.optimizers.Adam(lr=0.005),loss='mean_squared_error',metrics=['mean_absolute_error'])'''
    '''rnn_model.summary()'''
    history = rnn_model.fit(data, epochs=epochs, validation_data=data_test, shuffle=True, verbose=2)

    prediction = rnn_model.predict(X_test)
    prediction = prediction[:, -1]
    prediction = renormalize(prediction, priceMin, priceMax)

    real = Y_test[:, -1]
    real = np.array(real)
    real = renormalize(real, priceMin, priceMax)

    x = test.index.get_level_values('Date')[refDay + preDay - 1:]
    xshift = x + timedelta(days=preDay)

    font = FontProperties(fname=r"c:\windows\fonts\msjhbd.ttc", size=35)
    matplotlib.rcParams.update({'font.size': 20})

    fig1 = plt.figure()
    fig1.set_size_inches(18.5, 10.5)
    plt.plot(x, prediction, label="predict", color='b')
    plt.plot(xshift, prediction, linestyle='dotted', label="predict after shift", color='royalblue')
    plt.plot(x, real, label="real", color='r')

    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.xticks(np.arange(x[0], x[-1], timedelta(days=60)), rotation=15)
    plt.xlabel('day')
    plt.ylabel('price')
    '''plt.ylim(0, 1)'''
    plt.legend(fontsize=20)
    plt.title(target + " predict", fontproperties=font)

    fig2 = plt.figure()
    fig2.set_size_inches(18.5, 10.5)
    plt.ylim([0, 0.5])
    plt.plot(history.history[loss[1]], label=loss[1])
    plt.plot(history.history['val_' + loss[1]], label='val ' + loss[1])
    plt.plot(history.history['loss'], label=' loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.title(target + " error", fontsize=25, fontproperties=font)
    plt.legend(fontsize=20)

    '''rnn_model.save(root+filepath+'model.h5')'''
    plot_model(rnn_model, to_file=root + filepath + 'model.png', show_shapes=True)
    fig1.savefig(root + filepath + 'predict.svg')
    fig2.savefig(root + filepath + 'mae_mse.svg')

    hist_df = pd.DataFrame(history.history)
    hist_csv_file = 'history.csv'
    with open(root + filepath + 'history.csv', mode='w') as f:
        hist_df.to_csv(f)

    fobj = open(root + filepath + 'log.txt', 'w')
    fobj.write('epoch: ' + str(epochs) + '\n')
    fobj.write('x_train shape: ' + str(X_train.shape) + '\n')
    fobj.write('x_test shape: ' + str(X_test.shape) + '\n')
    fobj.write('training accuracy: ' + str(history.history['loss'][-1]) + '\n')
    fobj.write('test accuracy: ' + str(history.history['val_loss'][-1]) + '\n')
    fobj.write('---------------------------------------------------------------------------\n')
    fobj.write('\n')
    fobj.close()





