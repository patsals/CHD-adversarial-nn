#!/bin/bash

# input test permutations below
# DEFAULT CONFIGURATIONS, SKIPPING ANY FLAGS APPLIES THE FOLLOWING:
# python3 model_test.py --lambda_tradeoff=0.1 --epochs=100 --learning_rate=0.001 --patience=10 --batch_size=32 --adv_model_type=logistic_regression &

# ADD ADDITIONAL TESTS BELOW

# python3 model_test.py --lambda_tradeoff=0.05 --epochs=100 --learning_rate=0.001 --patience=10 --batch_size=32 --adv_model_type=svm &
# python3 model_test.py --lambda_tradeoff=0.1 --epochs=100 --learning_rate=0.001 --patience=10 --batch_size=32 --adv_model_type=svm &
# python3 model_test.py --lambda_tradeoff=0.3 --epochs=100 --learning_rate=0.001 --patience=10 --batch_size=32 --adv_model_type=svm &
# python3 model_test.py --lambda_tradeoff=0.05 --epochs=100 --learning_rate=0.01 --patience=10 --batch_size=32 --adv_model_type=svm &
# python3 model_test.py --lambda_tradeoff=0.1 --epochs=100 --learning_rate=0.01 --patience=10 --batch_size=32 --adv_model_type=svm &
# python3 model_test.py --lambda_tradeoff=0.3 --epochs=100 --learning_rate=0.01 --patience=10 --batch_size=32 --adv_model_type=svm &
# python3 model_test.py --lambda_tradeoff=0.05 --epochs=100 --learning_rate=0.1 --patience=10 --batch_size=32 --adv_model_type=svm &
# python3 model_test.py --lambda_tradeoff=0.1 --epochs=100 --learning_rate=0.1 --patience=10 --batch_size=32 --adv_model_type=svm &
# python3 model_test.py --lambda_tradeoff=0.3 --epochs=100 --learning_rate=0.1 --patience=10 --batch_size=32 --adv_model_type=svm &
# python3 model_test.py --lambda_tradeoff=0.05 --epochs=100 --learning_rate=0.001 --patience=10 --batch_size=64 --adv_model_type=svm &
# python3 model_test.py --lambda_tradeoff=0.1 --epochs=100 --learning_rate=0.001 --patience=10 --batch_size=64 --adv_model_type=svm &
# python3 model_test.py --lambda_tradeoff=0.3 --epochs=100 --learning_rate=0.001 --patience=10 --batch_size=64 --adv_model_type=svm &
# python3 model_test.py --lambda_tradeoff=0.05 --epochs=100 --learning_rate=0.01 --patience=10 --batch_size=64 --adv_model_type=svm &
# python3 model_test.py --lambda_tradeoff=0.1 --epochs=100 --learning_rate=0.01 --patience=10 --batch_size=64 --adv_model_type=svm &
# python3 model_test.py --lambda_tradeoff=0.3 --epochs=100 --learning_rate=0.01 --patience=10 --batch_size=64 --adv_model_type=svm &
# python3 model_test.py --lambda_tradeoff=0.05 --epochs=100 --learning_rate=0.1 --patience=10 --batch_size=64 --adv_model_type=svm &
# python3 model_test.py --lambda_tradeoff=0.1 --epochs=100 --learning_rate=0.1 --patience=10 --batch_size=64 --adv_model_type=svm &
# python3 model_test.py --lambda_tradeoff=0.3 --epochs=100 --learning_rate=0.1 --patience=10 --batch_size=64 --adv_model_type=svm &
# python3 model_test.py --lambda_tradeoff=0.05 --epochs=100 --learning_rate=0.001 --patience=10 --batch_size=32 --adv_model_type=logistic_regression &
# python3 model_test.py --lambda_tradeoff=0.1 --epochs=100 --learning_rate=0.001 --patience=10 --batch_size=32 --adv_model_type=logistic_regression &
# python3 model_test.py --lambda_tradeoff=0.3 --epochs=100 --learning_rate=0.001 --patience=10 --batch_size=32 --adv_model_type=logistic_regression &
# python3 model_test.py --lambda_tradeoff=0.05 --epochs=100 --learning_rate=0.01 --patience=10 --batch_size=32 --adv_model_type=logistic_regression &
# python3 model_test.py --lambda_tradeoff=0.1 --epochs=100 --learning_rate=0.01 --patience=10 --batch_size=32 --adv_model_type=logistic_regression &
# python3 model_test.py --lambda_tradeoff=0.3 --epochs=100 --learning_rate=0.01 --patience=10 --batch_size=32 --adv_model_type=logistic_regression &
# python3 model_test.py --lambda_tradeoff=0.05 --epochs=100 --learning_rate=0.1 --patience=10 --batch_size=32 --adv_model_type=logistic_regression &
# python3 model_test.py --lambda_tradeoff=0.1 --epochs=100 --learning_rate=0.1 --patience=10 --batch_size=32 --adv_model_type=logistic_regression &
# python3 model_test.py --lambda_tradeoff=0.3 --epochs=100 --learning_rate=0.1 --patience=10 --batch_size=32 --adv_model_type=logistic_regression &
# python3 model_test.py --lambda_tradeoff=0.05 --epochs=100 --learning_rate=0.001 --patience=10 --batch_size=64 --adv_model_type=logistic_regression &
# python3 model_test.py --lambda_tradeoff=0.1 --epochs=100 --learning_rate=0.001 --patience=10 --batch_size=64 --adv_model_type=logistic_regression &
# python3 model_test.py --lambda_tradeoff=0.3 --epochs=100 --learning_rate=0.001 --patience=10 --batch_size=64 --adv_model_type=logistic_regression &
# python3 model_test.py --lambda_tradeoff=0.05 --epochs=100 --learning_rate=0.01 --patience=10 --batch_size=64 --adv_model_type=logistic_regression &
# python3 model_test.py --lambda_tradeoff=0.1 --epochs=100 --learning_rate=0.01 --patience=10 --batch_size=64 --adv_model_type=logistic_regression &
# python3 model_test.py --lambda_tradeoff=0.3 --epochs=100 --learning_rate=0.01 --patience=10 --batch_size=64 --adv_model_type=logistic_regression &
# python3 model_test.py --lambda_tradeoff=0.05 --epochs=100 --learning_rate=0.1 --patience=10 --batch_size=64 --adv_model_type=logistic_regression &
# python3 model_test.py --lambda_tradeoff=0.1 --epochs=100 --learning_rate=0.1 --patience=10 --batch_size=64 --adv_model_type=logistic_regression &
# python3 model_test.py --lambda_tradeoff=0.3 --epochs=100 --learning_rate=0.1 --patience=10 --batch_size=64 --adv_model_type=logistic_regression &
python3 model_test.py --lambda_tradeoff=0.05 --epochs=100 --learning_rate=0.001 --patience=10 --batch_size=32 --adv_model_type=perceptron &
python3 model_test.py --lambda_tradeoff=0.1 --epochs=100 --learning_rate=0.001 --patience=10 --batch_size=32 --adv_model_type=perceptron &
python3 model_test.py --lambda_tradeoff=0.3 --epochs=100 --learning_rate=0.001 --patience=10 --batch_size=32 --adv_model_type=perceptron &
python3 model_test.py --lambda_tradeoff=0.05 --epochs=100 --learning_rate=0.01 --patience=10 --batch_size=32 --adv_model_type=perceptron &
python3 model_test.py --lambda_tradeoff=0.1 --epochs=100 --learning_rate=0.01 --patience=10 --batch_size=32 --adv_model_type=perceptron &
python3 model_test.py --lambda_tradeoff=0.3 --epochs=100 --learning_rate=0.01 --patience=10 --batch_size=32 --adv_model_type=perceptron &
python3 model_test.py --lambda_tradeoff=0.05 --epochs=100 --learning_rate=0.1 --patience=10 --batch_size=32 --adv_model_type=perceptron &
python3 model_test.py --lambda_tradeoff=0.1 --epochs=100 --learning_rate=0.1 --patience=10 --batch_size=32 --adv_model_type=perceptron &
python3 model_test.py --lambda_tradeoff=0.3 --epochs=100 --learning_rate=0.1 --patience=10 --batch_size=32 --adv_model_type=perceptron &
python3 model_test.py --lambda_tradeoff=0.05 --epochs=100 --learning_rate=0.001 --patience=10 --batch_size=64 --adv_model_type=perceptron &
python3 model_test.py --lambda_tradeoff=0.1 --epochs=100 --learning_rate=0.001 --patience=10 --batch_size=64 --adv_model_type=perceptron &
python3 model_test.py --lambda_tradeoff=0.3 --epochs=100 --learning_rate=0.001 --patience=10 --batch_size=64 --adv_model_type=perceptron &
python3 model_test.py --lambda_tradeoff=0.05 --epochs=100 --learning_rate=0.01 --patience=10 --batch_size=64 --adv_model_type=perceptron &
python3 model_test.py --lambda_tradeoff=0.1 --epochs=100 --learning_rate=0.01 --patience=10 --batch_size=64 --adv_model_type=perceptron &
python3 model_test.py --lambda_tradeoff=0.3 --epochs=100 --learning_rate=0.01 --patience=10 --batch_size=64 --adv_model_type=perceptron &
python3 model_test.py --lambda_tradeoff=0.05 --epochs=100 --learning_rate=0.1 --patience=10 --batch_size=64 --adv_model_type=perceptron &
python3 model_test.py --lambda_tradeoff=0.1 --epochs=100 --learning_rate=0.1 --patience=10 --batch_size=64 --adv_model_type=perceptron &
python3 model_test.py --lambda_tradeoff=0.3 --epochs=100 --learning_rate=0.1 --patience=10 --batch_size=64 --adv_model_type=perceptron

# ADDITIONAL NOTES:
# VALUE OPTIONS FOR "adv_model_type": ['perceptron', 'svm', 'logistic_regression']
