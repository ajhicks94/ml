import xml
import lxml.sax

def clean_and_count(article, data):
    for token in article.text.split():
        if token in data.keys():
            data[token] += 1
        else:
            data[token] = 1

########## SAX FOR STREAM PARSING ##########
class HyperpartisanNewsParser(xml.sax.ContentHandler):
    def __init__(self, mode, word_index={}, data=[]):
        xml.sax.ContentHandler.__init__(self)
        self.mode = mode
        self.lxmlhandler = "undefined"
        self.data = data
        self.word_index = word_index
        self.counter = 0
        self.left_count = 0
        self.right_count = 0
        self.neutral_count = 0

    def startElement(self, name, attrs):
        if name != "articles":
            if name == "article":
                self.lxmlhandler = lxml.sax.ElementTreeContentHandler()

            self.lxmlhandler.startElement(name, attrs)

    def characters(self, data):
        if self.lxmlhandler != "undefined":
            self.lxmlhandler.characters(data)

    def endElement(self, name):
        if self.lxmlhandler != "undefined":
            self.lxmlhandler.endElement(name)
            if name == "article":
                if self.mode == "widx":
                    clean_and_count(self.lxmlhandler.etree.getroot(), self.data)
                elif self.mode == "x":

                    article = self.lxmlhandler.etree.getroot()

                    row = []
 
                    # Split into sequence of words
                    textcleaned = article.text.split()
                    #print("Before:\n\n")
                    #print(textcleaned)
                    # Look up each word's index in freq index and append
                    for word in textcleaned:
                        try:
                            idx = self.word_index[word]
                            # Basically doing index_from in here
                            # so we can avoid errors on OOV_chars
                            idx += 3
                        except KeyError:
                            # OOV_CHAR
                            idx = 2
                        row.append(idx)
                    
                    # Append to sequence array
                    self.data.append(row)

                elif self.mode == "y":
                    article = self.lxmlhandler.etree.getroot()

                    bias = article.get('bias')

                    if (bias == 'left'):
                        self.left_count += 1
                    elif (bias == 'right'):
                        self.right_count += 1
                    else:
                        self.neutral_count += 1

                    hp = article.get('hyperpartisan')

                    if hp in ['true', 'True', 'TRUE']:
                        self.data.append(1)
                    elif hp in ['false', 'False', 'FALSE']:
                        self.data.append(0)
                    else:
                        err = "Mislabeled or unlabeled data found: " + hp
                        raise Exception(err)
                    
                self.counter += 1
                self.lxmlhandler = "undefined"

    def endDocument(self):
        if self.mode == 'y':
            print("Total count:", self.counter)
            print("\tleft:\t\t", self.left_count)
            print("\tright:\t\t", self.right_count)
            print("\tneutral:\t", self.neutral_count)