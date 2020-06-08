import mlflow
import os

if __name__ == '__main__':
    mlflow.log_param('param1',10)

    mlflow.log_metric('A', 10)
    mlflow.log_metric('B', 20)
    mlflow.log_metric('AC', 30)

    with open("output.txt", 'w') as f:
        f.write("Here be some output")
    mlflow.log_artifact('output.txt')