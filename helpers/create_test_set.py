import sys

from dataset_splitter import trim, write_trees
from count_articles import count

if __name__ == "__main__":
    num = 75000 #needs to be 75000
    tr = 'data/training_publisher/articles-training-bypublisher-20181122-cleaned.xml'
    #tr = sys.argv[1]
    tr_labels = 'data/gt_training_publisher/ground-truth-training-bypublisher-20181122.xml'
    #tr_labels = sys.argv[2]
    val = 'data/validation_publisher/articles-validation-bypublisher-20181122.xml'
    #val = sys.argv[3]
    val_labels = 'data/gt_validation_publisher/ground-truth-validation-bypublisher-20181122.xml'
    #val_labels = sys.argv[4]
    te = 'data/test/test.xml'
    #te = sys.argv[5]
    te_labels = 'data/test/test_labels.xml'
    #te_labels = sys.argv[6]

    d = ['app', tr_labels, val_labels, te_labels]

    print("=====Prior=====")
    #count(d)

    # 75000 articles training => test
    print("\nExtracting " + str(num) + " articles from training into test...")
    src_tree, src_label_tree, dest_tree, dest_label_tree = trim(tr, tr_labels, te, te_labels, num)
    write_trees(tr, tr_labels, te, te_labels, src_tree, src_label_tree, dest_tree, dest_label_tree)

    # Pull from the modified training set and not the original
    tr = 'data/training/training.xml'
    tr_labels = 'data/training/training_labels.xml'
    d = ['app', tr_labels, val_labels, te_labels]

    print("=====After " + str(num) + " tr -> te=====")
    count(d)

    # 75000 articles validation => test
    print("Extracting " + str(num) + " articles from validation into test")
    src_tree, src_label_tree, dest_tree, dest_label_tree = trim(val, val_labels, te, te_labels, num)
    write_trees(val, val_labels, te, te_labels, src_tree, src_label_tree, dest_tree, dest_label_tree)

    # Write to the modified validation set, not the original
    val = 'data/validation/validation.xml'
    val_labels = 'data/validation/validation_labels.xml'
    d = ['app', tr_labels, val_labels, te_labels]

    print("=====After " + str(num) + " val -> te=====")
    count(d)

    # 75000 articles training => validation
    print("Extracting " + str(num) + " articles from training into validation")
    src_tree, src_label_tree, dest_tree, dest_label_tree = trim(tr, tr_labels, val, val_labels, num)
    write_trees(tr, tr_labels, val, val_labels, src_tree, src_label_tree, dest_tree, dest_label_tree)

    print("=====After " + str(num) + " tr -> val")
    count(d)