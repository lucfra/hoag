import tensorflow as tf
import rfho as rf
import numpy as np


def rtho_experiment(datasets, saver=None, optimizer=rf.MomentumOptimizer, alpha0=0.,
                    hyper_batch_size=10, use_regularizer=True,
                    append_string='_RTHO', optimizer_kwargs=None,
                    hyper_optimizer_class=rf.GradientDescentOptimizer,
                    hyper_iterations=500,
                    hyper_optimizer_kwargs=None):
    if saver:
        saver.save_setting(vars(), excluded=optimizer_kwargs, append_string=append_string)

    train = datasets.train
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    model = rf.LinearModel(x, train.dim_data, train.dim_target)

    w, out, weights = rf.vectorize_model(model.var_list, model.out, model.Ws[0],  # weight matrix
                                         augment=optimizer.get_augmentation_multiplier())
    error = tf.reduce_sum(tf.log(1. + tf.exp(- y * out)))

    # generic_errors = [rf.binary_cross_entropy(o, d.target) for d, o in zip(datasets, outs)]

    alpha = tf.Variable(alpha0, name='alpha')

    training_error = error
    # training_error = tf.reduce_mean(generic_errors[0])
    if use_regularizer:
        training_error += tf.exp(alpha) * tf.reduce_sum(weights ** 2)
        # training_error += tf.exp(alpha) * tf.reduce_mean(w ** 2)
    # validation_error = tf.reduce_mean(generic_errors[1])
    # sum_val_err = tf.reduce_sum(generic_errors[1])
    # sum_tst_err = tf.reduce_sum(generic_errors[2])
    # test_error = tf.reduce_sum(generic_errors[2])

    # accuracies
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.sign(out), y), "float"))

    # validation_error = tf.reduce_sum

    dynamics = optimizer.create(w, loss=training_error, **optimizer_kwargs or {})
    hyper_opt = rf.HyperOptimizer(dynamics,
                                  {error: [alpha] + dynamics.get_optimization_hyperparameters(only_variables=True)
                                   }, method=rf.ForwardHG,
                                  hyper_optimizer_class=hyper_optimizer_class,
                                  **hyper_optimizer_kwargs or {})

    constraints = dynamics.get_natural_hyperparameter_constraints()

    trs = datasets.train.create_supplier(x, y)
    vls = datasets.validation.create_supplier(x, y)
    tss = datasets.test.create_supplier(x, y)

    if saver: saver.add_items('training error', training_error, trs,
                              'validation error (sum)', error, vls,
                              'test error (sum - validation in HOAG)', error, tss,
                              'validation accuracy', accuracy, vls,
                              'test accuracy', accuracy, tss,
                              'norm of weights', lambda stp: np.linalg.norm(
            w.var_list(rf.VlMode.TENSOR)[0].eval()),
                              # 'weights', w.var_list(rf.Vl_Mode.TENSOR)[0],
                              # 'out', out, val_sup,
                              *rf.flatten_list(
                                  [rf.simple_name(hyp), [hyp, hyper_opt.hyper_gradients.hyper_gradients_dict[hyp]]]
                                  for hyp in hyper_opt.hyper_list))

    with tf.Session(config=rf.CONFIG_GPU_GROWTH).as_default():
        hyper_opt.initialize()
        for k in range(hyper_iterations):
            if saver: saver.save(k, append_string=append_string)
            hyper_opt.run(hyper_batch_size,
                          train_feed_dict_supplier=trs,
                          val_feed_dict_suppliers={error: vls},
                          hyper_constraints_ops=constraints)
            # print(test_error.eval(), validation_error.eval(), training_error.eval(),
            #       [h.eval() for h in hyper_opt.hyper_list], sep='\t')
            # print()
    return saver.pack_save_dictionaries(append_string=append_string) if saver else None
