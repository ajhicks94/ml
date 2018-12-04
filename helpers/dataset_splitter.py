import lxml.etree as etree
import sys

def trim(src, src_labels, dest, dest_labels, num):
    #parser = etree.XMLParser(encoding="unicode", recover=True)
    #src_tree = etree.parse(src, parser)
    #if src == 'data/training/training.xml':
        #src_tree = etree.parse(src, parser)
    #else:
    #with open(src) as s:
        #src_tree = etree.parse(s, etree.XMLParser(encoding="utf-8"))
    src_tree = etree.parse(src)
    print(src_tree.docinfo.encoding)
    src_articles = src_tree.findall('.//article')

    src_label_tree = etree.parse(src_labels)#, parser)
    src_labels = src_label_tree.findall('.//article')

    dest_tree = etree.parse(dest)#, parser)
    dest_label_tree = etree.parse(dest_labels)#, parser)
    dest_label_tree_root = dest_label_tree.getroot()
    dest_tree_root = dest_tree.getroot()

    count = 0
    left_count = 0
    right_count = 0
    neutral_count = 0

    for article, label in zip(src_articles, src_labels):
        if left_count == num/4 and right_count == num/4 and neutral_count == num/2:
            return (src_tree, src_label_tree, dest_tree, dest_label_tree)
        else:
            b = label.get('bias')

            if left_count == num/4 and b == 'left':
                continue
            elif right_count == num/4 and b == 'right':
                continue
            elif neutral_count == num/2 and b not in ['left', 'right']:
                continue
            else:
                if b == 'left':
                    left_count += 1
                elif b == 'right':
                    right_count += 1
                else:
                    neutral_count += 1

                # Append to dest labels
                dest_label_tree_root.append(label)

                # Append to dest data
                dest_tree_root.append(article)

                count += 1

def write_trees(src, src_labels, dest, dest_labels, src_tree, src_label_tree, dest_tree, dest_label_tree):
    src_path = src.split('/')
    src_name = src_path[1].split('_')[0]
    src_prefix = src_path[0] + '/' + src_name + '/'

    # Given: data/training_publisher/somefilename.xml
    # Output: data/training/training_labels.xml
    #       : data/training/training.xml
    src_label_path = src_prefix + src_name + '_labels.xml'
    src_path = src_prefix + src_name + '.xml'

    dest_path = dest.split('/')
    dest_name = dest_path[1].split('_')[0]
    dest_prefix = dest_path[0] + '/' + dest_name + '/'

    dest_path = dest_prefix + dest_name + '.xml'
    dest_label_path = dest_prefix + dest_name + '_labels.xml'

    src_path = src + '.new'
    src_label_path = src_labels + '.new'
    dest_path = dest# + '.new'
    dest_label_path = dest_labels# + '.new'

    print("Writing new source file: " + src_path)
    print("Old source file: " + src)
    #with open(src_path, 'w') as f:
    #    f.write(etree.tostring(src_tree))
    #st = etree.tostring(src_tree, xml_declaration=True, encoding='UTF-8')
    #with open(src_path, 'wb') as f:
        #src_tree.write(f, encoding='unicode')
    try:
        src_tree.write(src_path, xml_declaration=False, encoding='UTF-8', standalone=False, pretty_print=False)
    except Exception:
        pass
    #src_tree.write(src_path, xml_declaration=False, encoding='UTF-8', standalone=False, method='xml', pretty_print=False)
    print("Writing new label file: " + src_label_path)
    src_label_tree.write(src_label_path, xml_declaration=False, standalone=False, encoding='UTF-8', pretty_print=True)

    print("Writing new dest file: " + dest_path)
    dest_tree.write(dest_path, xml_declaration=False, standalone=False, encoding='UTF-8', pretty_print=True)
    print("Writing new dest label file: " + dest_label_path)
    dest_label_tree.write(dest_label_path, xml_declaration=False, standalone=False, encoding='UTF-8', pretty_print=True)

    print("=====Data=====")
    print("count(//article):\t", dest_tree.xpath('count(//article)'))
    print("Num of articles left in src: ", src_tree.xpath('count(//article)'))

    print("\n=====Labels=====")
    print("count(//article):\t", dest_label_tree.xpath('count(//article)'))
    print("Num articles left in src: ", src_label_tree.xpath('count(//article)'))

if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Incorrect number of parameters")
        print("Usage: python dataset_splitter.py <source> <destination>")
        sys.exit()

    src = sys.argv[1]
    src_labels = sys.argv[2]
    dest = sys.argv[3]
    dest_labels = sys.argv[4]

    num_articles = sys.argv[5]

    src_tree, src_label_tree, dest_tree, dest_label_tree = trim(src, src_labels, dest, dest_labels, int(num_articles))
    
    # Write updated trees to file
    write_trees(src, src_labels, dest, dest_labels, src_tree, src_label_tree, dest_tree, dest_label_tree)