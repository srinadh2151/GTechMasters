""""""  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
template for generating data to fool learners (c) 2016 Tucker Balch  		  	   		 	 	 			  		 			     			  	 
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
  		  	   		 	 	 			  		 			     			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		 	 	 			  		 			     			  	 
GT User ID: snidadana3 (replace with your User ID)  		  	   		 	 	 			  		 			     			  	 
GT ID: 903966341 (replace with your GT ID)  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import math  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import numpy as np  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
# this function should return a dataset (X and Y) that will work  		  	   		 	 	 			  		 			     			  	 
# better for linear regression than decision trees  		  	   		 	 	 			  		 			     			  	 
def best_4_lin_reg(seed=1489683273):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		 	 	 			  		 			     			  	 
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		 	 	 			  		 			     			  	 
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param seed: The random seed for your data generation.  		  	   		 	 	 			  		 			     			  	 
    :type seed: int  		  	   		 	 	 			  		 			     			  	 
    :return: Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		 	 	 			  		 			     			  	 
    :rtype: numpy.ndarray  		  	   		 	 	 			  		 			     			  	 
    """
    np.random.seed(seed)
    # Generate a dataset with a linear relationship
    x = np.random.rand(100, 5)  # 100 rows, 5 columns
    y = np.dot(x, np.array([1.5, -2.0, 3.0, 0.5, -1.0])) + np.random.normal(0, 0.1, 100)

    return x, y

  		  	   		 	 	 			  		 			     			  	 
def best_4_dt(seed=1489683273):
    """
    Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		 	 	 			  		 			     			  	 
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		 	 	 			  		 			     			  	 
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param seed: The random seed for your data generation.  		  	   		 	 	 			  		 			     			  	 
    :type seed: int  		  	   		 	 	 			  		 			     			  	 
    :return: Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		 	 	 			  		 			     			  	 
    :rtype: numpy.ndarray  		  	   		 	 	 			  		 			     			  	 
    """    
    np.random.seed(seed)
    # # Generate a dataset with a non-linear relationship
    x = np.random.random((100, 2))
    y = []
    for i in range(x.shape[0]):
        a, b = x[i, 0], x[i, 1]
        if a > 0.5 and b > 0.5:
            y.append(150 * np.sin(a * np.pi) + 100 * np.cos(b * np.pi))
        elif a > 0.5:
            y.append(100 * np.sin(a * np.pi) - 50 * np.cos(b * np.pi))
        elif b > 0.5:
            y.append(50 * np.sin(a * np.pi) + 200 * np.cos(b * np.pi))
        else:
            y.append(-100 * np.sin(a * np.pi) - 150 * np.cos(b * np.pi))
    Y = np.array(y)
    
    return x, Y


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "snidadana3"  # Change this to your user ID

			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    print("they call me Srinadh.")
