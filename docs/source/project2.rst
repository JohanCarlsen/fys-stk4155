Project 2
=========

Artificial neural network class 
-------------------------------

.. autoclass:: ffnn.NeuralNetwork

Artificial neural network methods
---------------------------------

.. automethod:: ffnn.NeuralNetwork.fit

.. automethod:: ffnn.NeuralNetwork.predict

.. automethod:: ffnn.NeuralNetwork.calculate_score
.. automethod:: ffnn.NeuralNetwork.get_score_evolution

Regression analysis class 
-------------------------

.. autoclass:: linreg.RegressionAnalysis

Regression analysis methods 
---------------------------

.. automethod:: linreg.RegressionAnalysis.set_params

.. automethod:: linreg.RegressionAnalysis.set_hyper_params
.. automethod:: linreg.RegressionAnalysis.run
.. automethod:: linreg.RegressionAnalysis.get_score_evol

Logistic regression class 
-------------------------

.. autoclass:: logreg.LogisticRegression

Logistic regression methods
---------------------------

.. automethod:: logreg.LogisticRegression.fit

.. automethod:: logreg.LogisticRegression.predict

Preprocessing of data
---------------------

.. autofunction:: preprocess.center

.. autofunction:: preprocess.norm_data_zero_one

Cost functions
--------------

.. autoclass:: cost_funcs.MeanSquaredError
    :members:

.. autoclass:: cost_funcs.CrossEntropy
    :members:

.. autoclass:: cost_funcs.LogLoss
    :members:

Solver optimizers
-----------------

.. autoclass:: solvers.Constant
    :members:

.. autoclass:: solvers.ADAM
    :members:


Activation functions
--------------------

.. autoclass:: activations.Linear
    :members:

.. autoclass:: activations.Sigmoid
    :members:

.. autoclass:: activations.ReLU
    :members: