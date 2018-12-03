#!/usr/bin/env python

import sys
from lxml import etree

def count(args):
    if len(args) < 4:
        print("\nIncorrect num of parameters.")
        print("\nUsage: ")
        print("python " + args[0] + " <training> <validation> <test>")
        return

    count_articles_ex = "count(//article)"
    left_articles_ex = 'count(//article[@bias="left"])'
    right_articles_ex = 'count(//article[@bias="right"])'

    tr = args[1]
    val = args[2]
    te = args[3]

    tr_tree = etree.parse(tr)
    val_tree = etree.parse(val)
    te_tree = etree.parse(te)

    tr_articles = tr_tree.xpath(count_articles_ex)
    val_articles = val_tree.xpath(count_articles_ex)
    te_articles = te_tree.xpath(count_articles_ex)

    tr_left = tr_tree.xpath(left_articles_ex)
    val_left = val_tree.xpath(left_articles_ex)
    te_left = te_tree.xpath(left_articles_ex)

    tr_right = tr_tree.xpath(right_articles_ex)
    val_right = val_tree.xpath(right_articles_ex)
    te_right = te_tree.xpath(right_articles_ex)

    print("\nTraining:")
    print("\tTotal:\t", tr_articles)
    print("\tleft:\t", tr_left)
    print("\tright:\t", tr_right)

    print("Validation:")
    print("\tTotal:\t", val_articles)
    print("\tleft:\t", val_left)
    print("\tright:\t", val_right)

    print("Test:")
    print("\tTotal:\t", te_articles)
    print("\tleft:\t", te_left)
    print("\tright:\t", te_right)

if __name__ == "__main__":
    count(sys.argv)