# Reference: https://machinelearningmastery.com/mcnemars-test-for-machine-learning/

from statsmodels.stats.contingency_tables import mcnemar

def calculate(ctable):
    # Caculate mcnemar test
    result = mcnemar(ctable, exact=True)

    # Summarize the finding
    print('stastic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))

    # interpret the p-value
    alpha = 0.05

    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0)')
    else:
        print('Different proportions of errors (reject H0)')