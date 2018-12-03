from lxml import etree

def clean():
    src = 'data/training_publisher/articles-training-bypublisher-20181122.xml'

    tree = etree.parse(src)
    root = tree.getroot()
    articles = tree.findall('//article')
    
    for article in articles:
        print(article.get('id') + ' complete.')
        for child in article.getchildren():
            if child.text:
                child.text = ''.join([c for c in child.text if ord(c) < 127])

    tree.write('data/training/tt.xml', encoding='utf-8', pretty_print=True)

if __name__ == "__main__":
    clean()