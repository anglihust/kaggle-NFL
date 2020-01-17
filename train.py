
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.models import Model,load_model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout, merge, Add, BatchNormalization
import warnings
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from data_process import *
warnings.filterwarnings("ignore")

# evaluation metric
def crps(y_true, y_pred):
    y_true = np.clip(np.cumsum(y_true, axis=1), 0, 1)
    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
    return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0])


class Metric(Callback):
    def __init__(self, model, callbacks, data):
        super().__init__()
        self.model = model
        self.callbacks = callbacks
        self.data = data

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_end(self, batch, logs=None):
        X_train, y_train = self.data[0][0], self.data[0][1]
        y_pred = self.model.predict(X_train)
        y_true = np.clip(np.cumsum(y_train, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        tr_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_train[-1].shape[0])
        tr_s = np.round(tr_s, 6)
        logs['tr_CRPS'] = tr_s

        X_valid, y_valid = self.data[1][0], self.data[1][1]

        y_pred = self.model.predict(X_valid)
        y_true = np.clip(np.cumsum(y_valid, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_valid[-1].shape[0])
        val_s = np.round(val_s, 6)
        logs['val_CRPS'] = val_s
        print('tr CRPS', tr_s, 'val CRPS', val_s)

        for callback in self.callbacks:
            callback.on_epoch_end(batch, logs)


def op_model(x_train, y_train, x_test, y_test):
    num_layer1 = {{choice([128, 256, 512])}}
    num_layer2 = {{choice([128, 256, 512])}}
    num_layer3 = {{choice([64, 128, 256])}}
    lrate = {{uniform(0, 1)}}
    #batch_size = {{choice([64, 128, 512, 1024])}}
    drate1 = {{uniform(0, 1)}}
    drate2 = {{uniform(0, 1)}}
    drate3 = {{uniform(0, 1)}}
    inputs = Input(shape=(x_train.shape[1],))
    x = Dense(num_layer1, activation='relu')(inputs)
    x = Dropout(drate1)(x)
    x = BatchNormalization()(x)
    x = Dense(num_layer2, activation='relu')(x)
    x = Dropout(drate2)(x)
    x = BatchNormalization()(x)
    x = Dense(num_layer3, activation='relu')(x)
    x = Dropout(drate3)(x)
    x = BatchNormalization()(x)
    output = Dense(199, activation='softmax')(x)
    model = Model(inputs, output)
    optimizer = Adam(lr=lrate, beta_1=0.9, beta_2=0.95, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[])
    checkpoint = []
    checkpoint.append(ReduceLROnPlateau(monitor='val_CRPS', patience=8, verbose=1, factor=0.5, min_lr=0.00001))
    checkpoint.append(EarlyStopping(monitor='val_CRPS', min_delta=0, patience=10, verbose=0, mode='auto'))
    model.fit(x=x_train, y=y_train, batch_size=16, epochs=120, verbose=1, callbacks=checkpoint,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    return {'loss': score, 'status': STATUS_OK, 'model': model}

def retrain_model(model,x_train, y_train, x_test, y_test):
    checkpoint = []
    checkpoint.append(ReduceLROnPlateau(monitor='val_CRPS', patience=8, verbose=1, factor=0.5, min_lr=0.00001))
    checkpoint.append(EarlyStopping(monitor='val_CRPS', min_delta=0, patience=10, verbose=0, mode='auto'))
    model.fit(x=x_train, y=y_train, batch_size=16, epochs=120, verbose=1, callbacks=checkpoint,
              validation_data=(x_test, y_test))
    y_pred  = model.predict(x_test)
    crps_eval=crps(y_test, y_pred)
    return model, crps_eval

def train():
        x_train, y_train, x_test, y_test,X,yards,y=data()
        feature_importance(x_train, y_train)
        x_train, y_train, x_test, y_test,X,yards,y= feature_importance_selection(x_train, y_train, x_test, y_test, X, yards,y)
        best_run, best_model = optim.minimize(model=op_model,
                                              data=data,
                                              algo=tpe.suggest,
                                              max_evals=50,
                                              trials=Trials())
        best_model.save('bayesian_optimized.h5')
        print("Best performing model chosen hyper-parameters:")
        print(best_run)
        pred=best_model.predict(x_test)
        eval = crps(y_test,pred)
        print("eval metric:")
        print(eval)

        model = load_model('bayesian_optimized.h5')
        losses = []
        models = []
        crps_csv = []
        kfold = KFold(5, random_state=10, shuffle=True)
        for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(yards)):
            tr_x, tr_y = X[tr_inds], y[tr_inds]
            val_x, val_y = X[val_inds], y[val_inds]
            re_model, crps_eval = retrain_model(model,tr_x, tr_y, val_x, val_y)
            models.append(re_model)
            print("the %d fold crps is %f" % ((k_fold + 1), crps_eval))
            crps_csv.append(crps_eval)
        return models

def predict(x_te,models):
    model_num = len(models)
    for k, m in enumerate(models):
        if k == 0:
            y_pred = m.predict(x_te, batch_size=1024)
        else:
            y_pred += m.predict(x_te, batch_size=1024)

    y_pred = y_pred / model_num
    return y_pred




#
#
# important = ['back_from_scrimmage', 'min_dist', 'max_dist', 'mean_dist', 'std_dist',
#              'def_min_dist', 'def_max_dist', 'def_mean_dist', 'def_std_dist', 'X',
#              'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir', 'YardLine']
#
# cat = ['back_oriented_down_field', 'back_moving_down_field']
#
# num = ['back_from_scrimmage', 'min_dist', 'max_dist', 'mean_dist', 'std_dist', 'def_min_dist', 'def_max_dist',
#        'def_mean_dist', 'def_std_dist',
#        'X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir', 'YardLine', 'Distance'] + ['fe1', 'fe5', 'fe7', 'fe8', 'fe10',
#                                                                                    'fe11']
# num = [i for i in num if i in important]
# print(len(cat))
# print(len(num))



####






# def model_396_1():
#     inputs = []
#     embeddings = []
#     for i in cat:
#         input_ = Input(shape=(1,))
#         embedding = Embedding(int(np.absolute(X[i]).max() + 1), 10, input_length=1)(input_)
#         embedding = Reshape(target_shape=(10,))(embedding)
#         inputs.append(input_)
#         embeddings.append(embedding)
#     input_numeric = Input(shape=(len(num),))
#     embedding_numeric = Dense(512, activation='relu')(input_numeric)
#     inputs.append(input_numeric)
#     embeddings.append(embedding_numeric)
#     x = Concatenate()(embeddings)
#     x = Dense(256, activation='relu')(x)
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     output = Dense(199, activation='softmax')(x)
#     model = Model(inputs, output)
#     return model




# n_splits = 5
# kf = GroupKFold(n_splits=n_splits)
# score = []
# for i_369, (tdx, vdx) in enumerate(kf.split(X, y, X['GameId'])):
#     print(f'Fold : {i_369}')
#     X_train, X_val, y_train, y_val = X.iloc[tdx], X.iloc[vdx], y[tdx], y[vdx]
#     #X_train = [np.absolute(X_train[i]) for i in cat] + [X_train[num]]
#     #X_val = [np.absolute(X_val[i]) for i in cat] + [X_val[num]]
#
#     def data():
#         return X_train, y_train, X_val, y_val
#
#
#     best_run, best_model = optim.minimize(model=op_model,
#                                           data=data,
#                                           algo=tpe.suggest,
#                                           max_evals=30,
#                                           trials=Trials())
#     best_model.save(f'keras_369_{i_369}.h5')

#     model = model_396_1()
#     model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[])
#     es = EarlyStopping(monitor='val_CRPS',
#                    mode='min',
#                    restore_best_weights=True,
#                    verbose=2,
#                    patience=5)
#     es.set_model(model)
#     metric = Metric(model, [es], [(X_train,y_train), (X_val,y_val)])
#     for i in range(1):
#         model.fit(X_train, y_train, verbose=False)
#     for i in range(1):
#         model.fit(X_train, y_train, batch_size=64, verbose=False)
#     for i in range(1):
#         model.fit(X_train, y_train, batch_size=128, verbose=False)
#     for i in range(1):
#         model.fit(X_train, y_train, batch_size=256, verbose=False)
#     model.fit(X_train, y_train, callbacks=[metric], epochs=100, batch_size=1024, verbose=False)
#     score_ = crps(y_val, model.predict(X_val))
#     model.save(f'keras_369_{i_369}.h5')
#     print(score_)
#     score.append(score_)


# print(np.mean(score))


# models = []
# for i in range(n_splits):
#     models.append(load_model(f'keras_369_{i}.h5'))

# for (test_df, sample_prediction_df) in iter_test:
#     basetable = create_features(test_df, deploy=True)
#     basetable = process_two(basetable)
#     basetable[num] = scaler.transform(basetable[num])
#     test_ = [np.absolute(basetable[i]) for i in cat] + [basetable[num]]

#     y_pred = np.mean([model.predict(test_) for model in models], axis=0)
#     y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1).tolist()[0]

#     preds_df = pd.DataFrame(data=[y_pred], columns=sample_prediction_df.columns)
#     env.predict(preds_df)

# env.write_submission_file()


#
# class MultiLGBMClassifier():
#     def __init__(self, resolution, params):
#         ## smoothing size
#         self.resolution = resolution
#         ## initiarize models
#         self.models = [LGBMClassifier(**params) for _ in range(resolution)]
#
#     def fit(self, x, y):
#         self.classes_list = []
#         for k in range(self.resolution):
#             ## train each model
#             self.models[k].fit(x, (y + k) // self.resolution)
#             ## (0,1,2,3,4,5,6,7,8,9) -> (0,0,0,0,0,1,1,1,1,1) -> (0,5)
#             classes = np.sort(list(set((y + k) // self.resolution))) * self.resolution - k
#             classes = np.append(classes, 999)
#             self.classes_list.append(classes)
#
#     def predict(self, x):
#         pred199_list = []
#         for k in range(self.resolution):
#             preds = self.models[k].predict_proba(x)
#             classes = self.classes_list[k]
#             pred199s = self.get_pred199(preds, classes)
#             pred199_list.append(pred199s)
#         self.pred199_list = pred199_list
#         pred199_ens = np.mean(np.stack(pred199_list), axis=0)
#         return pred199_ens
#
#     def _get_pred199(self, p, classes):
#         ## categorical prediction -> predicted distribution whose length is 199
#         pred199 = np.zeros(199)
#         for k in range(len(p)):
#             pred199[classes[k] + 99: classes[k + 1] + 99] = p[k]
#         return pred199
#
#     def get_pred199(self, preds, classes):
#         pred199s = []
#         for p in preds:
#             pred199 = np.cumsum(self._get_pred199(p, classes))
#             pred199 = pred199 / np.max(pred199)
#             pred199s.append(pred199)
#         return np.vstack(pred199s)
#
# params = {'lambda_l1': 0.001, 'lambda_l2': 0.001,
#  'num_leaves': 40, 'feature_fraction': 0.4,
#  'subsample': 0.4, 'min_child_samples': 10,
#  'learning_rate': 0.01,
#  'num_iterations': 700, 'random_state': 42}
#
# model = MultiLGBMClassifier(resolution=5, params=params)
# model.fit(x_train, y_train)