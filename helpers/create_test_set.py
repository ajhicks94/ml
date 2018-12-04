import sys

from dataset_splitter import trim, write_trees
from count_articles import count

def create_small_set(tr, tr_labels, val, val_labels, te, te_labels, total):

    small_tr = tr + '.small'
    small_tr_labels = tr_labels + '.small'
    small_val = val + '.small'
    small_val_labels = val_labels + '.small'

    tr_size = int(total * 0.6)
    val_size = int(total * 0.2)
    te_size = int(val_size)

    d = ['app', tr_labels, val_labels, te_labels]

    #print("=====Prior=====")
    #count(d)

    # 60-20-20 SPLIT

    # te_size / 2 articles to test from tr
    print("Sending", te_size / 2, "articles from training to test")
    src_tree, src_label_tree, dest_tree, dest_label_tree = trim(tr, tr_labels, te, te_labels, te_size/2)
    write_trees(tr, tr_labels, te, te_labels, src_tree, src_label_tree, dest_tree, dest_label_tree)

    # Pull from the modified training set and not the original
    tr = tr + '.new'
    tr_labels = tr_labels + '.new'
    d = ['app', tr_labels, val_labels, te_labels]

    #count(d)

    # te_size / 2 articles to test from val
    print("Sending", te_size/2, "articles from validation to test")
    src_tree, src_label_tree, dest_tree, dest_label_tree = trim(val, val_labels, te, te_labels, te_size/2)
    write_trees(val, val_labels, te, te_labels, src_tree, src_label_tree, dest_tree, dest_label_tree)

    # Write to the modified validation set, not the original
    val = val + '.new'
    val_labels = val_labels + '.new'
    d = ['app', tr_labels, val_labels, te_labels]

    #count(d)

    # tr_size articles to small_tr from tr
    print("Sending", tr_size, "articles from training to small_training")
    src_tree, src_label_tree, dest_tree, dest_label_tree = trim(tr, tr_labels, small_tr, small_tr_labels, tr_size)
    write_trees(tr, tr_labels, small_tr, small_tr_labels, src_tree, src_label_tree, dest_tree, dest_label_tree)

    # val_size articles to small_val from val
    if val_size != 100000:
        print("Sending", val_size, "articles from validation to small_validation")
        src_tree, src_label_tree, dest_tree, dest_label_tree = trim(val, val_labels, small_val, small_val_labels, val_size)
        write_trees(val, val_labels, small_val, small_val_labels, src_tree, src_label_tree, dest_tree, dest_label_tree)
        val = 'data/gt_validation_publisher/ground-truth-validation-bypublisher-20181122.xml.small'
    else:
        val = 'data/gt_validation_publisher/ground-truth-validation-bypublisher-20181122.xml.new'

    tr = 'data/gt_training_publisher/ground-truth-training-bypublisher-20181122.xml.small'
    te = 'data/test_publisher/test_labels.xml'

    count(['app', tr, val, te])

if __name__ == "__main__":
    num = 75000 #needs to be 75000
    total = sys.argv[7]
    #tr = 'data/training_publisher/articles-training-bypublisher-20181122.xml'
    tr = sys.argv[1]
    #tr_labels = 'data/gt_training_publisher/ground-truth-training-bypublisher-20181122.xml'
    tr_labels = sys.argv[2]
    #val = 'data/validation_publisher/articles-validation-bypublisher-20181122.xml'
    val = sys.argv[3]
    #val_labels = 'data/gt_validation_publisher/ground-truth-validation-bypublisher-20181122.xml'
    val_labels = sys.argv[4]
    #te = 'data/test/test.xml'
    te = sys.argv[5]
    #te_labels = 'data/test/test_labels.xml'
    te_labels = sys.argv[6]

    d = ['app', tr_labels, val_labels, te_labels]

    create_small_set(tr, tr_labels, val, val_labels, te, te_labels, int(total))
    '''
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
    '''