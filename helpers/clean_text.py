import sys
import re

from lxml import etree

if __name__ == "__main__":
    filename = sys.argv[1]

    # Parse the xml file
    print("Parsing the file...")
    tree = etree.parse(filename)

    # Strip unnecessary tags AND attach them directly to their parent article :)
    print("Stripping away p, a, and q tags...")
    etree.strip_tags(tree, 'p', 'a', 'q')

    # Clean away any unneeded chars, including unicode! :D
    print("Remove characters: [^a-z ]")
    for article in tree.getroot().getchildren():
        article.text = re.sub('[^a-z ]', '', article.text.lower())

    # Save xml file
    # Unknown serialization error happens
    # But all the articles are still intact
    print("Writing to file...")
    try:
        tree.write(filename + '.cleaned')
    except Exception:
        pass

    print("Done!")