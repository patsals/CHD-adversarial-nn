import pandas as pd

epochs = [100]
batch_size = [32, 64]
patience = [10]
learning_rate = [0.001, 0.01, 0.1]
lambda_tradeoff = [0.05, 0.1, 0.3]
adv_model_type = ["svm", "logistic_regression", "perceptron"]


df = pd.DataFrame(columns=['epochs', 'batch_size', 'patience', 'learning_rate', 'lambda_tradeoff', 'adv_model_type'])

tests = []
for amt in adv_model_type:
    for e in epochs:
        for bs in batch_size:
            for p in patience:
                for lr in learning_rate:
                    for lt in lambda_tradeoff:
                        df.loc[len(df)] = {
                            'epochs': e,
                            'batch_size': bs,
                            'patience': p,
                            'learning_rate': lr,
                            'lambda_tradeoff': lt,
                            'adv_model_type': amt
                        }

                        tests.append(
                            f'python3 model_test.py --lambda_tradeoff={lt} --epochs={e} --learning_rate={lr} --patience={p} --batch_size={bs} --adv_model_type={amt}'
                        )


df = df.sort_values(by=['adv_model_type', 'epochs', 'batch_size', 'patience', 'learning_rate', 'lambda_tradeoff'])

df.to_csv('parameter_permutations.csv', index=False)


with open("tests.txt", "w") as file:
    file.writelines(f"{item}\n" for item in tests) 