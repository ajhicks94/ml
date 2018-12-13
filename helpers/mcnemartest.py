# Reference: https://machinelearningmastery.com/mcnemars-test-for-machine-learning/

from statsmodels.stats.contingency_tables import mcnemar

def get_c_table(predict1, predict2):
    yy = 0
    yn = 0
    ny = 0
    nn = 0
    with open(predict1, 'r') as f:
        with open(predict2, 'r') as r:
            for l1, l2 in zip(f,r):
                ll1 = l1.rstrip('\n').split()
                ll2 = l2.rstrip('\n').split()

                if ll1[1] == "true" and ll2[1] == "true":
                    yy += 1
                elif ll1[1] == "true" and ll2[1] == "false":
                    yn += 1
                elif ll1[1] == "false" and ll2[1] == "true":
                    ny += 1
                elif ll1[1] == "false" and ll2[1] == "false":
                    nn += 1
    
    return [[yy,yn],[ny,nn]]

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