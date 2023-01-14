import numpy as np
import data_handler as dh
import models
from tensorflow.keras.optimizers import Adam


def learn(model, train_batch, train_targets):    
    acc = 0.0
    loss = 0.0
    i = 0
    for batch in zip(train_batch, train_targets):
        data, target = batch

        batch_metrics = model.train_on_batch(data, target)

        acc += batch_metrics[1]
        loss += batch_metrics[0]
        i+=1

    acc = (acc/i)*100
    loss = (loss/i)
    return acc, loss


def evaluate(model, test_batch, test_targets):
    correct = 0
    i = 0
    for batch in zip(test_batch, test_targets):
        data, target = batch

        probs = model.predict_on_batch(data)
        
        if np.argmax(probs) == np.argmax(target):
            correct +=1
        i+=1

    acc = (correct/i)*100
    return acc


def experiment(cfg, real_data, reference_data, zero_shot_cls=None):

    # data setting
    train_dict, test_dict = dh.train_test_split(real_data, cfg.split_ratio)
    test_batch, test_targets = dh.get_test_batch(test_dict, reference_data, ref_key=cfg.ref_key, zero_shot_cls=zero_shot_cls)

    # model & hyperparameters setting
    input_shape = test_batch[0][0][0].shape                # (length, points)

    if cfg.model_type == 'HybridBaseModel':
        base_model = models.HybridBaseModel(input_shape)
    else:
        base_model = models.ViTBaseModel(input_shape, cfg.ViT_params)

    optimizer = Adam(learning_rate = cfg.lr)
    model = models.binary_siamese_net(input_shape, base_model.model())
    model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ['acc'])

    # learn & evaluate
    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    for e in range(cfg.epochs):

        train_batch, train_targets = dh.get_train_batch(train_dict, cfg.batch_size, cfg.n_batch, reference_data, ref_key=cfg.ref_key, zero_shot_cls=zero_shot_cls)
        train_acc, train_loss = learn(model, train_batch, train_targets)
        test_acc = evaluate(model, test_batch, test_targets)

        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)

        print('epoch : {}, train acc : {:.4f} %, train loss : {:.8f}, test acc : {:.4f} %, '.format(e+1, train_acc, train_loss, test_acc))
    
    return model, train_acc_list, train_loss_list, test_acc_list
