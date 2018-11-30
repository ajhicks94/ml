import lxml.etree as etree  #import etree, like c's include
import sys

def trim(xml, max_articles):
    #tree = etree.fromstring(xml)
    tree = etree.parse(xml)
    articles = tree.findall('.//article')

    count = 0
    for article in articles:
        # Remove all articles after we reach maximum amount desired
        # Later, change this so that we can choose to redirect the rest to a new xml file?
        print(article.get('id'))
        if count >= max_articles:
            print("removed article ", article.get('id'))
            print("count= ", count)
            article.getparent().remove(article)
        count += 1

    return tree

if __name__ == "__main__":
    s = r'''<xml>
    <articles>
        <article id="5">
            <somelabel>
            </somelabel>
        </article>
        <article id="1">
            <stff>
            </stff>
        </article>
    </articles>
    </xml>'''

    filename = sys.argv[1]
    max_articles = sys.argv[2]
    #res = trim(s, int(max_articles))
    newtree = trim(filename, int(max_articles))
    
    # Not printing first line= <?xml version="1.0" encoding="UTF-8" standalone="no"?>
    with open(sys.argv[1], 'w') as f:
        f.write(etree.tostring(newtree, encoding='unicode', pretty_print=True))

    #print(etree.tostring(res))  #and you have your edited xml.