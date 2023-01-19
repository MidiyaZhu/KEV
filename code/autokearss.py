import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import autokeras as ak

trainSet = pd.read_csv('emotion-labels-train.csv')
valSet = pd.read_csv('emotion-labels-val.csv')
testSet = pd.read_csv('emotion-labels-test.csv')
trainSet.head()

SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 100, 110, 123, 124, 52, 342, 543, 120, 1234, 543, 76, 657]
train = pd.concat([trainSet, valSet], axis=0).reset_index(drop=True)
for seed in SEEDS:
# Split data
        X_train, X_val, y_train, y_val = train_test_split(train['text'], train['label'],
                                                            stratify=train['label'],
                                                            test_size=0.2, random_state=seed)

        # X_train_ak = np.array(trainSet['text'])
        # y_train_ak = np.array(trainSet['label'])
        # X_val_ak = np.array(valSet['text'])
        # y_val_ak = np.array(valSet['label'])
        X_test_ak = np.array(testSet['text'])
        y_test_ak = np.array(testSet['label'])


        # Preparing the data for autokeras
        X_train_ak = np.array(X_train)
        y_train_ak = np.array(y_train)
        X_val_ak = np.array(X_val)
        y_val_ak = np.array(y_val)

        node_input = ak.TextInput()
        node_output = ak.TextToIntSequence()(node_input)
        node_output = ak.Embedding()(node_output)
        node_output = ak.ConvBlock(separable=True)(node_output)
        node_output = ak.ClassificationHead()(node_output)
        keras = ak.AutoModel(inputs=node_input, outputs=node_output, overwrite=True, max_trials=3)

        # Fit the training dataset
        keras.fit(X_train_ak, y_train_ak, epochs=20, validation_data=(X_val_ak,y_val_ak))

        # keras_export = keras.export_model()
        # keras_export.summary()

        pred_keras = keras.predict(X_test_ak)
        testaccuracy=accuracy_score(y_test_ak, pred_keras)
        # Compute the accuracy
        print('Accuracy: ' + str(accuracy_score(y_test_ak, pred_keras)))
        f1, macro, micro = f1_score(y_test_ak, pred_keras, average='weighted'), f1_score(y_test_ak, pred_keras, average='macro'), f1_score(
                y_test_ak, pred_keras, average='micro')
        print('f1: ', f1, '\nmacro: ', macro,'\nmicro: ',micro )
      
