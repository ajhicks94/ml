mv data/gt_training_publisher/ground-truth-training-bypublisher-20181122.xml.small data/training/large/train_labels.small
mv data/gt_validation_publisher/ground-truth-validation-bypublisher-20181122.xml.small data/validation/large/val_labels.small
mv data/training_publisher/articles-training-bypublisher-20181122.xml.cleaned.small data/training/large/train.small
mv data/validation_publisher/articles-validation-bypublisher-20181122.xml.cleaned.small data/validation/large/val.small
mv data/test_publisher/test.xml data/test/large/test.small
mv data/test_publisher/test_labels.xml data/test/large/test_labels.small

rm data/training_publisher/*.new
rm data/validation_publisher/*.new
rm data/test_publisher/*.new
rm data/gt_training_publisher/*.new
rm data/gt_validation_publisher/*.new
rm data/training_publisher/*.small
rm data/validation_publisher/*.small
rm data/test_publisher/*.small
rm data/gt_training_publisher/*.small
rm data/gt_validation_publisher/*.small


cp data/gt_test_publisher/blank.xml data/test_publisher/test.xml
cp data/gt_test_publisher/blank.xml data/test_publisher/test_labels.xml
cp data/gt_test_publisher/blank.xml data/training_publisher/articles-training-bypublisher-20181122.xml.cleaned.small
cp data/gt_test_publisher/blank.xml data/gt_training_publisher/ground-truth-training-bypublisher-20181122.xml.small
cp data/gt_test_publisher/blank.xml data/validation_publisher/articles-validation-bypublisher-20181122.xml.cleaned.small
cp data/gt_test_publisher/blank.xml data/gt_validation_publisher/ground-truth-validation-bypublisher-20181122.xml.small