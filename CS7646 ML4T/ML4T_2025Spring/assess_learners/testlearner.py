""""""  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			     			  	 
All Rights Reserved  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			     			  	 
or edited.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			     			  	 
GT honor code violation.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	 	 			  		 			     			  	 
"""
	 			  		 			     			  	 
import math
import sys
import os
import numpy as np
import pandas as pd
import LinRegLearner as lrl
import DTLearner as dtl
import RTLearner as rtl
import BagLearner as bl
import InsaneLearner as it
import matplotlib.pyplot as plt

def save_plot(fig, title):
    # if not os.path.exists('images'):
    #     os.makedirs('images')
    fig.savefig(f'images/{title}.png')


def experiment_1(filename):
    inf = open(filename)
    df = pd.read_csv(inf)
    data = df.drop(columns=['date']).to_numpy() if 'date' in df.columns else df.to_numpy()

    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    leaf_sizes = range(1, 51)
    in_sample_rmse = []
    out_sample_rmse = []

    for leaf_size in leaf_sizes:
        learner = dtl.DTLearner(leaf_size=leaf_size, verbose=False)
        learner.add_evidence(train_x, train_y)

        pred_y_in = learner.query(train_x)
        rmse_in = math.sqrt(((train_y - pred_y_in) ** 2).sum() / train_y.shape[0])
        in_sample_rmse.append(rmse_in)

        pred_y_out = learner.query(test_x)
        rmse_out = math.sqrt(((test_y - pred_y_out) ** 2).sum() / test_y.shape[0])
        out_sample_rmse.append(rmse_out)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(leaf_sizes, in_sample_rmse, label='In-sample RMSE')
    plt.plot(leaf_sizes, out_sample_rmse, label='Out-of-sample RMSE')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('Overfitting Analysis with DTLearner')
    plt.legend()
    plt.grid(True)
    plt.show()
    save_plot(fig, 'experiment_1')


def experiment_2(filename):
    inf = open(filename)
    df = pd.read_csv(inf)
    data = df.drop(columns=['date']).to_numpy() if 'date' in df.columns else df.to_numpy()

    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    leaf_sizes = range(1, 51)
    in_sample_rmse = []
    out_sample_rmse = []

    for leaf_size in leaf_sizes:
        learner = bl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size": leaf_size}, bags=20, boost=False, verbose=False)
        learner.add_evidence(train_x, train_y)

        pred_y_in = learner.query(train_x)
        rmse_in = math.sqrt(((train_y - pred_y_in) ** 2).sum() / train_y.shape[0])
        in_sample_rmse.append(rmse_in)

        pred_y_out = learner.query(test_x)
        rmse_out = math.sqrt(((test_y - pred_y_out) ** 2).sum() / test_y.shape[0])
        out_sample_rmse.append(rmse_out)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(leaf_sizes, in_sample_rmse, label='In-sample RMSE with Bagging')
    plt.plot(leaf_sizes, out_sample_rmse, label='Out-of-sample RMSE with Bagging')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('Bagging Effect on Overfitting with DTLearner')
    plt.legend()
    plt.grid(True)
    plt.show()
    save_plot(fig, 'experiment_2')


def experiment_3(filename):
    inf = open(filename)
    df = pd.read_csv(inf)
    data = df.drop(columns=['date']).to_numpy() if 'date' in df.columns else df.to_numpy()

    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    leaf_sizes = range(1, 51)
    dt_in_sample_mae = []
    rt_in_sample_mae = []

    for leaf_size in leaf_sizes:
        dt_learner = dtl.DTLearner(leaf_size=leaf_size, verbose=False)
        rt_learner = rtl.RTLearner(leaf_size=leaf_size, verbose=False)

        dt_learner.add_evidence(train_x, train_y)
        rt_learner.add_evidence(train_x, train_y)

        dt_pred_y_in = dt_learner.query(train_x)
        rt_pred_y_in = rt_learner.query(train_x)

        dt_mae_in = np.mean(np.abs(train_y - dt_pred_y_in))
        rt_mae_in = np.mean(np.abs(train_y - rt_pred_y_in))

        dt_in_sample_mae.append(dt_mae_in)
        rt_in_sample_mae.append(rt_mae_in)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(leaf_sizes, dt_in_sample_mae, label='DTLearner In-sample MAE')
    plt.plot(leaf_sizes, rt_in_sample_mae, label='RTLearner In-sample MAE')
    plt.xlabel('Leaf Size')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Comparison of DTLearner and RTLearner')
    plt.legend()
    plt.grid(True)
    plt.show()
    save_plot(fig, 'experiment_3')



# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python testlearner.py <filename> <experiment>")
#         sys.exit(1)
    
#     filename = sys.argv[1]
#     experiment = sys.argv[2]

    # if experiment == "experiment_1":
    #     experiment_1(filename)
    # elif experiment == "experiment_2":
    #     experiment_2(filename)
    # elif experiment == "experiment_3":
    #     experiment_3(filename)  
    #     print(f"Unknown experiment: {experiment}")
    #     sys.exit(1)

def test_learner(learner_name, filename):
        inf = open(filename)
        df= pd.read_csv(inf)
        data = df.drop(columns=['date']).to_numpy() if 'date' in df.columns else df.to_numpy()
        print('\nColumn names are \n', df.columns)
        # data = np.array(
        #     [list(map(float, s.strip().split(","))) for s in inf.readlines()]
        # )

        # compute how much of the data is training and testing
        train_rows = int(0.6 * data.shape[0])
        test_rows = data.shape[0] - train_rows

        # separate out training and testing data
        train_x = data[:train_rows, 0:-1]
        train_y = data[:train_rows, -1]
        test_x = data[train_rows:, 0:-1]
        test_y = data[train_rows:, -1]

        print(f"{test_x.shape}")
        print(f"{test_y.shape}")

        # create a learner based on the input argument
        if learner_name == "LinRegLearner":
            learner = lrl.LinRegLearner(verbose=True)
        elif learner_name == "DTLearner":
            learner = dtl.DTLearner(leaf_size=1, verbose=True)
        elif learner_name == "RTLearner":
            learner = rtl.RTLearner(leaf_size=1, verbose=True)
        elif learner_name == "BagLearner":
            learner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=True)
        else:
            print(f"Unknown learner: {learner_name}")
            sys.exit(1)

        # train the learner
        learner.add_evidence(train_x, train_y)
        print(learner.author())

        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        print()
        print("In sample results")
        print(f"RMSE: {rmse}")
        c = np.corrcoef(pred_y, y=train_y)
        print(f"corr: {c[0,1]}")

        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        print()
        print("Out of sample results")
        print(f"RMSE: {rmse}")
        c = np.corrcoef(pred_y, y=test_y)
        print(f"corr: {c[0,1]}")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    # learner_name = sys.argv[2]
    # experiment = sys.argv[3]
    # if experiment == "experiment_1":

    for learner_name in ["LinRegLearner", "DTLearner", "RTLearner", "BagLearner"]:
        print(learner_name, filename, ' - started \n')
        test_learner(learner_name, filename)
        print(learner_name, filename, ' - completed \n')


    experiments = {"experiment_1": experiment_1, "experiment_2": experiment_2, "experiment_3": experiment_3}
    for exp_name, experiment in experiments.items():
        print(exp_name, filename, ' - started \n')
        experiment(filename)
        print(exp_name, filename, ' - completed \n')



