import sys

from dataset_splitter import trim, write_trees

if __name__ == "__main__":
    tr = sys.argv[1]
    val = sys.argv[2]
    te = sys.argv[3]

    # 75000 articles training => test
    src_tree, dest_tree = trim(tr, te, 75000)
    write_trees(tr, te, src_tree, dest_tree)

    # 75000 articles validation => test
    src_tree, dest_tree = trim(val, te, 75000)
    write_trees(val, te, src_tree, dest_tree)

    # 75000 articles training => validation
    src_tree, dest_tree = trim(tr, val, 75000)
    write_trees(tr, val, src_tree, dest_tree)