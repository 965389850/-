import numpy as np
import os
from keras.models import load_model
import pandas as pd
import tensorflow as tf
import trainlog.buildtrainlog as tlog

import sys 
sys.path.insert(1, './testmodel')
import model.IResNet_BiLSTM as mod


# # ---------------导入UNSW-NB15数据集---------------------
# # x_train 训练集数据            x_test 测试集数据
# # y_train2 训练集2分类标签       y_test2 测试集2分类标签
# # y_train5 训练集5分类标签       y_test5 测试集5分类标签
# unswpath = './testmodel/data/UNSW-NB15/data/'
# x_train = pd.read_csv(unswpath+'UNSW_x_train.csv')
# x_test = pd.read_csv(unswpath+'UNSW_x_test.csv')
# y_train5 = pd.read_csv(unswpath+'UNSW_y_train.csv')
# y_train2 = y_train5['label']
# y_train5 = y_train5['attack']
# y_test5 = pd.read_csv(unswpath+'UNSW_y_test.csv')
# y_test2 = y_test5['label']
# y_test5 = y_test5['attack']


#-----------------导入CICIDS2017数据集------------------
cicpath = './testmodel/data/CICIDS2017/'
x_train = pd.read_csv(cicpath+'x_train.csv')
x_test = pd.read_csv(cicpath+'x_test.csv')
y_train15 = pd.read_csv(cicpath+'y_train_new.csv')
y_test15 = pd.read_csv(cicpath+'y_test_new.csv')


#--------------参数---------------
# -----------logpath--------------
logpath = './testmodel/modeltrain/trainlog/IRLGtrainlog.csv'

#---------paramname,paramrange-----------
# 输入参数：参数名、参数调整范围
# 参数名：
paramname = ['test1','test2','test3']
# 参数调整范围：
range1 = [32,64,128,256]
range2 = [32,64,128,256]
range3 = [32,64,128,256]
# paramrange = [range1, range2, range3, range4]

#----------checkpoint save root --------------
checkpoint_save_root = './testmodel/save/IRLG/'

# ------------X_train Y_train X_test Y_test--------------
Y_train = y_train15['binary']
Y_test = y_test15['binary']


def adjustparam(logpath = logpath, paramname = paramname, range1  = range1, range2 = range2, range3 = range3, checkpoint_save_root = checkpoint_save_root):
    # 生成trainlog文件
    if not os.path.exists(logpath):
        tlog.buildtrainlog(paramname, range1, range2, range3, logpath)

    log = pd.read_csv(logpath)
    print('reading log doc:')
    print(log.head())
    index = log[log['ok'].isnull()].index
    num = index[0]
    while num <= log.index[-1]:
        data = log.loc[num]
        print('getting log data:\n',data)
        param = []
        for i in range(0,len(paramname)):
            param.append(data[paramname[i]])
        print('getting params',param)

        checkpoint_save_path = checkpoint_save_root+'IRLG5_'+str(param[0])+'_'+str(param[1])+'_'+str(param[2])+'_.h5'

        if not os.path.exists(checkpoint_save_path):
            print('modle is not exists, training!')
            #------------- model ---------------
            model = mod.new_IRLG(int(param[0]),int(param[1]),int(param[2]),mod = 'softmax2')
            # model = mod.IResNet_BiLSTM(int(param[0]),int(param[1]),int(param[2]),mod = 'softmax2')
            cp_callback = [tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_save_path,
                                                            monitor = 'val_sparse_categorical_accuracy',
                                                            save_best_only=True
                                                            ),
                        #    tf.keras.callbacks.EarlyStopping(
                        #                                     monitor = 'val_sparse_categorical_accuracy'
                        #                                     # monitor = 'val_categorical_accuracy'
                        #                                     # monitor = 'binary_accuracy'
                        #                                     # monitor = 'val_accuracy'
                        #                                     # monitor = 'accuracy'
                        #                                     ,patience=5,min_delta=0.001,mode='max'
                        #                                     ),
                            tf.keras.callbacks.ReduceLROnPlateau(
                                monitor = 'val_sparse_categorical_accuracy'
                                ,factor = 0.5
                                ,patience=3
                            )
            ]

            model.compile(optimizer=tf.keras.optimizers.Adam(
                                                                learning_rate=0.001
                                                                # learning_rate=1e-05
                                                                # beta_1=0.9,beta_2=0.99
                                                            ),               #训练时选择哪种优化器
                            loss=
                            # tf.keras.losses.BinaryCrossentropy(),
                            tf.keras.losses.SparseCategoricalCrossentropy(),
                            # tf.keras.losses.CategoricalCrossentropy(),

                            metrics=[
                                # 'binary_accuracy'
                                # 'accuracy'
                                # tf.keras.metrics.CategoricalAccuracy()
                                # km.categorical_accuracy()
                                tf.keras.metrics.SparseCategoricalAccuracy()
                                # km.sparse_categorical_recall(),
                                # km.sparse_categorical_f1_score()
                                # tf.keras.metrics.CategoricalAccuracy()
                                ]) 

            history = model.fit(x_train, Y_train, batch_size=1000, epochs=20
                            # ,validation_split=0.20
                            ,validation_data=(x_test,Y_test)
                            ,validation_freq=1
                            ,callbacks=cp_callback
                            ,shuffle=True)

            model.summary()

            loss = history.history['loss']
        else:
            print('model is exists, loading')
            loss = [1]

        from sklearn.metrics import classification_report,accuracy_score,f1_score, precision_score, recall_score, confusion_matrix
        model = load_model(checkpoint_save_path)
        y_pred = model.predict(x_test)  #, batch_size=64, verbose=1
        y_pred_bool = np.argmax(y_pred, axis=1)
        # print(y_pred_bool)

        a = classification_report(Y_test, y_pred_bool)
        print(a)

        acc = accuracy_score(Y_test, y_pred_bool)
        pre = precision_score(Y_test, y_pred_bool, average="macro")
        rec = recall_score(Y_test, y_pred_bool , average="macro")
        f1 = f1_score(Y_test, y_pred_bool , average="macro")
        print("acc=",acc)
        print("precision=",pre)
        print("recall=",rec)
        print("f1=",f1)


        data['Loss'] = loss[-1]
        data['Acc'] = acc
        data['Precision'] = pre
        data['Recall'] = rec
        data['F1'] = f1


        import time
        t=time.localtime()
        ok = time.strftime("%Y-%m-%d %H:%M:%S",t)
        print(time.strftime("%Y-%m-%d %H:%M:%S",t))
        data['ok'] = ok

        print('updating log data')
        log.loc[num] = data
        print('saving log document')
        log.to_csv(logpath,index=False)
        # print('saving model')
        # model.save('./save/IRABL5/IRABL5_'+str(i)+'_'+str(j)+'_'+str(k)+'_'+str(l)+'.h5')


        num += 1

        del(model)

if __name__ == '__main__':
    adjustparam()