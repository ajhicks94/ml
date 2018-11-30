#!/usr/bin/env python

import sys
from lxml import etree

if __name__ == "__main__":
    filename = sys.argv[1]

    tree = etree.parse(filename)
    num_articles = tree.xpath("count(//article)")
    print(num_articles)