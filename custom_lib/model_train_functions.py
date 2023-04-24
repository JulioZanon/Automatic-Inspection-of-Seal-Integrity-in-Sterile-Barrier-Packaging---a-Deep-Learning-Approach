from __future__ import absolute_import, division, print_function, unicode_literals
from custom_lib.model_build_funtions import compute_class_freqs, get_BinaryCrossentropy_weighted
import tensorflow as tf
import os
layers = tf.keras.layers
import keras_tuner
#from tensorflow.python.summary.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
#from tensorboard.backend.event_processing import event_accumulator
from custom_lib.model_build_funtions import soft_dice_loss


def get_compiler_param(lr = 0.0001, optimiser_function = 'RMSprop',momentum1 = 0.85, momentum2 = 0.95, loss_function = 'BinaryCrossentropy', class_freqs = []):

    # Extract optimiser inputs
    momentum = momentum1
    Exp_decay_rate_momentum1 = momentum1
    Exp_decay_rate_momentum2 = momentum2

    # Select Optimiser Function
    if optimiser_function == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, rho=0.9, momentum=momentum, global_clipnorm=1.0)
    elif optimiser_function == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=Exp_decay_rate_momentum1, beta_2=Exp_decay_rate_momentum2, global_clipnorm=1.0)
    elif optimiser_function == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=True, global_clipnorm=1.0)
    else:
        raise RuntimeError('Selected optimiser function: [' + optimiser_function + '] is not supported.')
    # Select Loss Function
    if loss_function == 'BinaryCrossentropy':
        loss = tf.keras.losses.BinaryCrossentropy() # from_logits=True
        metrics = ['acc']
    elif loss_function == 'BinaryCrossentropy_weighted':
        neg_weights, pos_weights = compute_class_freqs(class_freqs)
        loss = get_BinaryCrossentropy_weighted(pos_weights, neg_weights)
        metrics = ['acc']
    elif loss_function == 'hinge':
        loss = tf.keras.losses.hinge(from_logits=True)
        metrics = ['acc']
    elif loss_function == 'categorical_crossentropy':
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        metrics = ['acc']
    elif loss_function == 'categorical_hinge':
        loss = tf.keras.losses.CategoricalHinge(from_logits=True)
        metrics = ['acc']
    elif loss_function == 'SparseCategoricalCrossentropy':
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ['acc']
    elif loss_function == 'soft_dice_loss':
        loss = soft_dice_loss
        metrics = ['acc']
    else: raise RuntimeError('Selected loss function: [' + loss_function + '] is not supported.')


    #Additional metrics to be added in future updates
    """METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]"""

    # Generate graph with optimiser and loss function
    return optimizer, loss, metrics


def compile_model(model, lr, momentum1, momentum2, optimiser_func, loss_func, class_freqs):

    # convert lists to values as this function could be called form optimisers
    if isinstance(lr, list): lr = lr[0]
    if isinstance(momentum1, list): momentum1 = momentum1[0]
    if isinstance(momentum2, list): momentum2 = momentum2[0]
    if isinstance(optimiser_func, list): optimiser_func = optimiser_func[0]
    if isinstance(loss_func, list): loss_func = loss_func[0]


    compiler_optimizer, compiler_loss, compiler_metrics = get_compiler_param(lr=lr, optimiser_function=optimiser_func, momentum1=momentum1, momentum2=momentum1,
                       loss_function=loss_func, class_freqs=class_freqs)


    model.compile(optimizer=compiler_optimizer, loss=compiler_loss, metrics=compiler_metrics)

    return model

def get_hyp_param_tuner_func(model, lr, momentum1, momentum2, optimiser_func, loss_func, class_freqs):
    def tuner_function(hp):
        momentum2_not_required = False
        hp_learning_rate = lr
        if isinstance(lr, list):
            if len(lr) == 4: hp_learning_rate = hp.Float('learning_rate', min_value=lr[1], max_value=lr[2], step=lr[3], default=lr[0])
            else: hp_learning_rate = lr[0]
        hp_optimiser_func = optimiser_func
        if isinstance(optimiser_func, list):
            if len(optimiser_func) > 1:  hp_optimiser_func = hp.Choice('optimiser_func', values=optimiser_func, default=optimiser_func[0])
            else: hp_optimiser_func = optimiser_func[0]
        if not isinstance(hp_optimiser_func, list):
            if hp_optimiser_func != 'Adam': momentum2_not_required = True
        hp_momentum1 = momentum1
        if isinstance(momentum1, list):
            if len(momentum1) == 4: hp_momentum1 = hp.Float('momentum_1', min_value=momentum1[1], max_value=momentum1[2], step=momentum1[3], default=momentum1[0])
            else: hp_momentum1 = momentum1[0]
        hp_momentum2 = momentum2
        if isinstance(momentum2, list):
            if len(momentum2) == 4 and not momentum2_not_required:  hp_momentum2 = hp.Float('momentum_2', min_value=momentum2[1], max_value=momentum2[2], step=momentum2[3], default=momentum2[0])
            else: hp_momentum2 = momentum2[0]
        hp_loss_func = loss_func
        if isinstance(loss_func, list):
            if len(loss_func) > 1:
                hp_loss_func = hp.Choice('loss_func', values=loss_func, default=loss_func[0])
            else: hp_loss_func = loss_func[0]


        compiler_optimizer, compiler_loss, compiler_metrics = get_compiler_param(lr=hp_learning_rate, optimiser_function=hp_optimiser_func,
                                                                                 loss_function= hp_loss_func, momentum1= hp_momentum1,
                                                                                 momentum2=hp_momentum2, class_freqs=class_freqs)

        model.compile(optimizer=compiler_optimizer, loss=compiler_loss, metrics=compiler_metrics)

        return model
    return tuner_function



class kt_BayesianOptimization(keras_tuner.tuners.BayesianOptimization):

  def run_trial(self, trial, *args, **kwargs):
    # You can add additional HyperParameters for preprocessing and custom training loops
    # via overriding `run_trial`
    epoch_range = [20, 10, 30, 5] # Standard: [20, 10, 30, 5] - Test: [2, 1, 2, 1]
    kwargs['epochs'] = trial.hyperparameters.Int('epochs',min_value=epoch_range[1], max_value=epoch_range[2], step=epoch_range[3], default=epoch_range[0])
    return super(kt_BayesianOptimization, self).run_trial(trial, *args, **kwargs)


class get_trial_planer:
    def __init__(self, model_trial_info={}, model_dataset_info={}, model_tune_param={}, search_plan=[]):
        self.model_trial_info = model_trial_info
        self.model_dataset_info = model_dataset_info
        self.model_tune_param = model_tune_param
        self.search_plan = search_plan
        self.num_iter = len(search_plan) + 1
        self.i = 0
    def __iter__(self):
        return self

    def __next__(self):
        if self.i > self.num_iter: raise StopIteration
        if self.i==0:
            model_trial_info = self.model_trial_info.copy()
            model_trial_info['trial_id'] = model_trial_info['trial_id'] + '_' + str(self.i)
            self.i += 1
            return model_trial_info, self.model_dataset_info, self.model_tune_param
        else:
            return self.replace_combination()

    def replace_combination(self):
        p_combination = self.search_plan[self.i - 1]

        model_trial_info = self.model_trial_info.copy()
        model_trial_info['trial_id'] = model_trial_info['trial_id'] + '_' + str(self.i)
        model_dataset_info = self.model_dataset_info.copy()
        model_tune_param = self.model_tune_param.copy()

        for key_p in p_combination.keys():
            for key in model_trial_info.keys():
                if key_p == key:
                    model_trial_info[key] = p_combination[key_p]
            for key in model_dataset_info.keys():
                if key_p == key:
                    model_dataset_info[key] = p_combination[key_p]
            for key in model_tune_param.keys():
                if key_p == key:
                    model_tune_param[key] = p_combination[key_p]
        self.i += 1
        return model_trial_info, model_dataset_info, model_tune_param


def extract_history_from_tuner(best_trial, logdir, trial_id):

    acc = []
    val_acc = []
    loss = []
    val_loss = []

    path_Training = os.path.join(logdir, trial_id, best_trial, 'execution0',  'train')
    ea = EventAccumulator(path_Training)
    ea.Reload()
    for i in range(len(ea.Tensors('epoch_loss'))):
        acc.append(float(tf.make_ndarray(ea.Tensors('epoch_acc')[i][2]).max()))
        loss.append(float(tf.make_ndarray(ea.Tensors('epoch_loss')[i][2]).max()))
        #lr.append(ea.Scalars('epoch_lr')[i][2])

    # for i in range(len(ea.Scalars('epoch_loss'))):
    #     acc.append(ea.Scalars('epoch_acc')[i][2])
    #     loss.append(ea.Scalars('epoch_loss')[i][2])
    #     #lr.append(ea.Scalars('epoch_lr')[i][2])


    path_Validation = os.path.join(logdir, trial_id, best_trial, 'execution0',  'validation')
    ea = EventAccumulator(path_Validation)
    ea.Reload()
    for i in range(len(ea.Tensors('epoch_loss'))):
      val_acc.append(float(tf.make_ndarray(ea.Tensors('epoch_acc')[i][2]).max()))
      val_loss.append(float(tf.make_ndarray(ea.Tensors('epoch_loss')[i][2]).max()))

    return acc, val_acc, loss, val_loss