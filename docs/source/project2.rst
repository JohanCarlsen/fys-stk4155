Project 2
=========

Artificial neural network class 
===============================

.. autoclass:: ffnn.NeuralNetwork

Artificial neural network methods
=================================

.. automethod:: ffnn.NeuralNetwork.fit

.. automethod:: ffnn.NeuralNetwork.predict

.. automethod:: ffnn.NeuralNetwork.calculate_score

Regression analysis class 
=========================

.. autoclass:: analysis.RegressionAnalysis

Regression analysis methods 
===========================

.. automethod:: analysis.RegressionAnalysis.set_params

.. automethod:: analysis.RegressionAnalysis.set_hyper_params
.. automethod:: analysis.RegressionAnalysis.run

Logistic regression class 

.. autoclass:: logreg.LogisticRegression

Logistic regression methods
===========================

.. automethod:: logreg.LogisticRegression.fit

.. automethod:: logreg.LogisticRegression.predict

Preprocessing of data
=====================

.. autofunction:: preprocess.center

.. autofunction:: preprocess.norm_data_zero_one

Cost functions
==============

.. autoclass:: cost_funcs.MeanSquaredError
    :members:

.. autoclass:: cost_funcs.CrossEntropy
    :members:

.. autoclass:: cost_funcs.LogLoss
    :members:

Activation functions
====================

.. autoclass:: activations.Linear
    :members:

.. autoclass:: activations.Sigmoid
    :members: