import strawberryfields as sf
from strawberryfields.ops import Dgate, BSgate, Rgate, Sgate, Kgate, Interferometer
import tensorflow as tf
from qmlt.tf.helpers import make_param
from qmlt.tf import CircuitLearner

import numpy as np
import pandas

depth = 1
steps = 500


def circuit(X):
    num_qubits = X.get_shape().as_list()[1]

    phi_1 = make_param(name='phi_1', stdev=np.sqrt(2) / num_qubits, shape=[depth, num_qubits], regularize=False)
    theta_1 = make_param(name='theta_1', stdev=np.sqrt(2) / num_qubits, shape=[depth, num_qubits], regularize=False)
    a = make_param(name='a', stdev=np.sqrt(2) / num_qubits, shape=[depth, num_qubits], regularize=False, monitor=True)
    rtheta_1 = make_param(name='rtheta_1', stdev=np.sqrt(2) / num_qubits, shape=[depth, num_qubits], regularize=False,
                          monitor=True)
    r = make_param(name='r', stdev=np.sqrt(2) / num_qubits, shape=[depth, num_qubits], regularize=False, monitor=True)
    kappa = make_param(name='kappa', stdev=np.sqrt(2) / num_qubits, shape=[depth, num_qubits], regularize=False,
                       monitor=True)
    phi_2 = make_param(name='phi_2', stdev=np.sqrt(2) / num_qubits, shape=[depth, num_qubits], regularize=False)
    theta_2 = make_param(name='theta_2', stdev=np.sqrt(2) / num_qubits, shape=[depth, num_qubits], regularize=False)
    rtheta_2 = make_param(name='rtheta_2', stdev=np.sqrt(2) / num_qubits, shape=[depth, num_qubits], regularize=False,
                          monitor=True)

    def layer(i, size):
        Rgate(rtheta_1[i, 0]) | (q[0])
        BSgate(phi_1[i, 0], 0) | (q[0], q[1])
        Rgate(rtheta_1[i, 2]) | (q[1])

        for j in range(size):
            Sgate(r[i, j]) | q[j]

        Rgate(rtheta_2[i, 0]) | (q[0])
        BSgate(phi_2[i, 0], 0) | (q[0], q[1])
        Rgate(rtheta_2[i, 2]) | (q[2])
        BSgate(phi_2[i, 2], theta_2[i, 3]) | (q[2], q[3])
        Rgate(rtheta_2[i, 1]) | (q[1])
        BSgate(phi_2[i, 1], 0) | (q[1], q[2])
        Rgate(rtheta_2[i, 0]) | (q[0])
        BSgate(phi_2[i, 0], 0) | (q[0], q[1])
        Rgate(rtheta_2[i, 0]) | (q[0])
        Rgate(rtheta_2[i, 1]) | (q[1])
        Rgate(rtheta_2[i, 2]) | (q[2])
        Rgate(rtheta_2[i, 3]) | (q[3])
        BSgate(phi_2[i, 2], 0) | (q[2], q[3])
        Rgate(rtheta_2[i, 2]) | (q[2])
        BSgate(phi_2[i, 1], 0) | (q[1], q[2])
        Rgate(rtheta_2[i, 1]) | (q[1])

        for j in range(size):
            Kgate(kappa[i, j]) | q[j]

    eng, q = sf.Engine(num_qubits)

    with eng:
        for i in range(num_qubits):
            Dgate(X[:, i], 0.) | q[i]
        for d in range(depth):
            layer(d, num_qubits)

    num_inputs = X.get_shape().as_list()[0]
    state = eng.run('tf', cutoff_dim=10, eval=False, batch_size=num_inputs)
    circuit_output, var0 = state.quad_expectation(0)

    return circuit_output


def myloss(circuit_output, targets):
    return tf.losses.mean_squared_error(labels=circuit_output, predictions=targets)


def myregularizer(regularized_params):
    return tf.nn.l2_loss(regularized_params)


def outputs_to_predictions(circuit_output):
    return circuit_output


df_train = pandas.read_csv('FFtrain2.csv', header=0)
df_test = pandas.read_csv('FFtest2.csv', header=0)

df_train = np.array(df_train)
df_test = np.array(df_test)

np.random.shuffle(df_train)
np.random.shuffle(df_test)

X_train = df_train[0:30, 0:4]
Y_train = df_train[0:30, 4]

X_test = df_test[0:10, 0:4]
Y_test = df_test[0:10, 4]

hyperparams = {'circuit': circuit,
               'task': 'supervised',
               'loss': myloss,
               'optimizer': 'SGD',
               # 'regularizer': myregularizer,
               # 'regularization_strength': 0.1,
               'init_learning_rate': 0.1
               }

learner = CircuitLearner(hyperparams=hyperparams)

num_train_inputs = X_train.shape[0]

learner.train_circuit(X=X_train, Y=Y_train, steps=50, batch_size=10)

test_score = learner.score_circuit(X=X_test, Y=Y_test,
                                   outputs_to_predictions=outputs_to_predictions)
print("\nPossible scores to print: {}".format(list(test_score.keys())))
print("Accuracy on test set: ", test_score['accuracy'])
print("Loss on test set: ", test_score['loss'])

outcomes = learner.run_circuit(X=X_test, outputs_to_predictions=outputs_to_predictions)

print("\nPossible outcomes to print: {}".format(list(outcomes.keys())))
print("Predictions for new inputs: {}".format(outcomes['predictions']))
print("Real outputs for new inputs: {}".format(Y_test))