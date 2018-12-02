import lxml.etree as etree
import sys

def trim(xml, max_articles):
    tree = etree.parse(xml)
    articles = tree.findall('.//article')
    secondtree = tree
    
    count = 0
    for article in articles:
        # Remove all articles after we reach maximum amount desired
        # Later, change this so that we can choose to redirect the rest to a new xml file?
        if count >= max_articles:
            article.getparent().remove(article)
        count += 1
    return tree

if __name__ == "__main__":

    filename = sys.argv[1]
    max_articles = sys.argv[2]
    newtree = trim(filename, int(max_articles))
    
    with open(sys.argv[1], 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + etree.tostring(newtree, encoding='unicode', pretty_print=True))