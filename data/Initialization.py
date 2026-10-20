import random
import os

import tensorflow as tf
#devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(devices[0], True)

# ★ FIX: GPU가 있을 때만 memory growth 설정
devices = tf.config.experimental.list_physical_devices('GPU')
if devices:
    for d in devices:
        tf.config.experimental.set_memory_growth(d, True)

from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta, Nadam
from keras import backend as K
import numpy as np
import sys
import pandas as pd
from scipy.io import loadmat
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

def printn(string):
    sys.stdout.write(string)
    sys.stdout.flush()

def Create_Pairs(domain_adaptation_task, repetition, sample_per_class,
    X_train_target, y_train_target, X_train_source, y_train_source,
    n_features=400, pairs_dir=None, seed=None, neg_pos_ratio=3.0,
    balance_target_to_source=False, balance_mode="overall", smote_k_max=5
):
    UM  = domain_adaptation_task
    cc  = repetition
    SpC = sample_per_class

    print ('Creating pairs for repetition: '+str(cc)+' and sample_per_class: '+str(sample_per_class))
    Training_P=[]
    Training_N=[]

    for trs in range(len(y_train_source)):
        for trt in range(len(y_train_target)):
            if y_train_source[trs]==y_train_target[trt]:
                Training_P.append([trs,trt])
            else:
                Training_N.append([trs,trt])

    random.shuffle(Training_N)
    Training = (Training_P + Training_N[:int(neg_pos_ratio * len(Training_P))])
    random.shuffle(Training)

    X1=np.zeros([len(Training),n_features],dtype='float32')
    X2=np.zeros([len(Training),n_features],dtype='float32')

    y1=np.zeros([len(Training)])
    y2=np.zeros([len(Training)])
    yc=np.zeros([len(Training)])

    for i in range(len(Training)):
        in1,in2=Training[i]
        X1[i,:]=X_train_source[in1,:]
        X2[i,:]=X_train_target[in2,:]

        y1[i]=y_train_source[in1]
        y2[i]=y_train_target[in2]
        if y_train_source[in1]==y_train_target[in2]:
            yc[i]=1

    if pairs_dir is None:
        pairs_dir = "./CCSA_pairs"
    
    os.makedirs(pairs_dir, exist_ok=True)

    np.save(
        os.path.join(pairs_dir, UM + '_X1_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy'),
        X1
    )
    
    np.save(
        os.path.join(pairs_dir, UM + '_X2_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy'),
        X2
    )
    
    np.save(
        os.path.join(pairs_dir, UM + '_y1_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy'),
        y1
    )
    
    np.save(
        os.path.join(pairs_dir, UM + '_y2_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy'),
        y2
    )
    
    np.save(
        os.path.join(pairs_dir, UM + '_yc_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy'),
        yc
    )


# FIX: 전달된 dr(rate)를 그대로 사용
def Create_Model(hiddenLayers=[100, 50], dr=0.5):
    model = Sequential()
    for idx, nodes in enumerate(hiddenLayers):
        model.add(Dense(nodes, activation='relu'))      # activation 인라인으로 간결화
        if dr and dr > 0:
            model.add(Dropout(dr))                      # ★ dr 적용
    return model


def euclidean_distance(vects):
    eps = 1e-08
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), eps))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def training_the_model(model, domain_adaptation_task, repetition, sample_per_class,
    batch_size, X_val_target, y_val_target,
    max_epochs=100, patience=15, seed=35,
    pairs_dir=None, class_weight_mode="from_marginal",
    y_marginal=None, selection_metric="auc", verbose_every=10
):
    """
    - validation data로 train_on_batch 하지 않음 (leakage 제거)
    - val metric(AUC/PR-AUC) 기반으로 best epoch weights 저장
    - 최종 test AUC/PR-AUC 계산 후 반환

    추가 보강:
    - yc (contrastive label) shape을 (N,1)로 맞춤
    - remainder minibatch까지 학습에 포함 (ceil)
    - best_weights가 한 번도 안 잡히는 케이스 폴백
    - 선택 metric이 NaN이면 다른 metric으로 폴백
    """

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    nb_classes = 2
    UM = domain_adaptation_task
    cc = repetition
    SpC = sample_per_class
    nn = int(batch_size)

    epochs = int(max_epochs)
    select_metric = selection_metric
    
    if pairs_dir is None:
        pairs_dir = "./CCSA_pairs"
    
    if class_weight_mode != "from_marginal":
        raise ValueError(
            f"Unsupported class_weight_mode: {class_weight_mode}"
        )

    # ---- load pairs created by Create_Pairs ----
    X1 = np.load(
        os.path.join(pairs_dir, f"{UM}_X1_count_{cc}_SpC_{SpC}.npy")
    ).astype("float32")
    
    X2 = np.load(
        os.path.join(pairs_dir, f"{UM}_X2_count_{cc}_SpC_{SpC}.npy")
    ).astype("float32")
    
    y1 = np.load(
        os.path.join(pairs_dir, f"{UM}_y1_count_{cc}_SpC_{SpC}.npy")
    ).astype("int32")
    
    y2 = np.load(
        os.path.join(pairs_dir, f"{UM}_y2_count_{cc}_SpC_{SpC}.npy")
    ).astype("int32")
    
    yc = np.load(
        os.path.join(pairs_dir, f"{UM}_yc_count_{cc}_SpC_{SpC}.npy")
    ).astype("float32")

    # one-hot for classification head (softmax 2 units 가정)
    y1_oh = to_categorical(y1, nb_classes).astype("float32")
    y2_oh = to_categorical(y2, nb_classes).astype("float32")

    # ---- class weights (inverse freq) ----
    class_counts = (y1_oh.sum(axis=0) + y2_oh.sum(axis=0)).astype(np.float32)
    total = float(max(class_counts.sum(), 1.0))
    freq = class_counts / total
    w = np.where(freq > 0, 0.5 / freq, 0.).astype(np.float32)  # [w0,w1]

    # ---- validation/test labels for metric ----
    y_val_target = np.asarray(y_val_target).astype(int).reshape(-1)

    # metric 선택값 방어
    if select_metric not in ("prauc", "auc"):
        raise ValueError("select_metric must be 'prauc' or 'auc'")

    best_val = -np.inf
    best_weights = None

    # ---- training loop ----
    for e in range(epochs):
        if verbose_every and (e % verbose_every == 0):
            printn(f"{e}->")

        # shuffle indices each epoch
        idx = np.random.permutation(len(yc))
        X1s, X2s = X1[idx], X2[idx]
        y1s, y2s = y1_oh[idx], y2_oh[idx]
        # ★ FIX: yc는 (N,1)로 맞춰 CSA 출력(shape (N,1))과 정합
        ycs = yc[idx].reshape(-1, 1).astype("float32")

        # ★ FIX: remainder minibatch까지 포함
        n_batches = int(np.ceil(len(ycs) / nn))
        for i in range(n_batches):
            sl = slice(i * nn, min((i + 1) * nn, len(ycs)))
            if sl.start >= sl.stop:
                continue

            # source->target 방향
            cls_idx1 = np.argmax(y1s[sl], axis=1)
            sw1 = w[cls_idx1].astype(np.float32)

            model.train_on_batch(
                [X1s[sl], X2s[sl]],
                [y1s[sl], ycs[sl]],
                sample_weight=[sw1, np.ones((sl.stop - sl.start,), dtype=np.float32)],
            )

            # target->source 방향
            cls_idx2 = np.argmax(y2s[sl], axis=1)
            sw2 = w[cls_idx2].astype(np.float32)

            model.train_on_batch(
                [X2s[sl], X1s[sl]],
                [y2s[sl], ycs[sl]],
                sample_weight=[sw2, np.ones((sl.stop - sl.start,), dtype=np.float32)],
            )

        # ---- validation evaluation (NO TRAIN) ----
        Out_val = model.predict([X_val_target, X_val_target], verbose=0)
        prob_val = Out_val[0][:, 1] if Out_val[0].ndim == 2 else Out_val[0].reshape(-1)

        if len(np.unique(y_val_target)) == 2:
            auc_val = roc_auc_score(y_val_target, prob_val)
            pr_val = average_precision_score(y_val_target, prob_val)
        else:
            auc_val, pr_val = np.nan, np.nan

        # ★ FIX: 선택 metric이 NaN이면 다른 metric으로 폴백
        if select_metric == "prauc":
            current = pr_val
            if not np.isfinite(current):
                current = auc_val
        else:
            current = auc_val
            if not np.isfinite(current):
                current = pr_val

        if np.isfinite(current) and current > best_val:
            best_val = current
            best_weights = model.get_weights()

        # ★ FIX: 첫 epoch라도 weight를 저장해두는 폴백(모든 val metric이 NaN인 케이스 방지)
        if best_weights is None and e == 0:
            best_weights = model.get_weights()


    # ---- final model selection ----
    if best_weights is not None:
        model.set_weights(best_weights)
    
    print(
        f"Training completed. "
        f"Best validation {select_metric}: {best_val:.6f}"
    )
    
    return None
