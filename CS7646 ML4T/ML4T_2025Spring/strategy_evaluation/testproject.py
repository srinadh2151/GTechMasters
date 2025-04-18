""""""
"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch

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

****** Unit tests for Strategy Evaluation ******

"""

import ManualStrategy
import StrategyLearner
import experiment1
import experiment2
import logging

def main():
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    
    # Execute Manual Strategy
    log.info("Running Manual Strategy...")
    ManualStrategy.main()

    # # Execute Strategy Learner
    # print("Running Strategy Learner...")
    # StrategyLearner.main()

    # Execute Experiment 1
    log.info("Running Experiment 1...")
    experiment1.main()

    # Execute Experiment 2
    log.info("Running Experiment 2...")
    experiment2.main()

if __name__ == '__main__':
    main()


