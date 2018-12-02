import lxml.etree as etree
import sys

def trim(src, dest, num):
    src_tree = etree.parse(src)
    src_articles = src_tree.findall('.//article')

    dest_tree = etree.parse(dest)
    dest_tree_root = dest_tree.getroot()

    count = 0
    left_count = 0
    right_count = 0
    neutral_count = 0
    
    for article in src_articles:
        if left_count == num/4 and right_count == num/4 and neutral_count == num/2:
            print("Total articles written: ", count, "==", num)
            print("\tleft:\t", left_count)
            print("\tright:\t", right_count)
            print("\tneutral:", neutral_count)
            print("count(//article):\t", dest_tree.xpath('count(//article)'))
            print("Number of articles left in src: ", src_tree.xpath('count(//article)'))
            return (src_tree, dest_tree)
        else:
            b = article.get('bias')

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

                # Insert into dest tree and remove from src_tree
                dest_tree_root.append(article)

                count += 1

def write_trees(src, dest, src_tree, dest_tree):
    dest_folder = dest.split('/')
    dest_folder = dest_folder[0] + '/' + dest_folder[1].split('_')[1] + '/' + dest_folder[1].split('_')[1] + '.xml'
    print(dest_folder)
    src_tree.write(src + '.temp', xml_declaration=True, standalone=False, method='xml', encoding='UTF-8', pretty_print=True)
    dest_tree.write(dest + '.temp', xml_declaration=True, standalone=False, method='xml', encoding='UTF-8', pretty_print=True)

if __name__ == "__main__":

    src = sys.argv[1]
    dest = sys.argv[2]

    num_articles = sys.argv[3]

    src_tree, dest_tree = trim(src, dest, int(num_articles))
    
    # Write updated trees to file
    write_trees(src, dest, src_tree, dest_tree)