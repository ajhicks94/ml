Source code
    - located at https://www.github.com/ajhicks94/ml on master branch

Datasets (Medium2 for debugging and Large2 for test, Full just for reference)
    - located at https://drive.google.com/drive/folders/1VUM7BHVFYejDPXbhgaHr9tQyRt9jfa7N?usp=sharing

Original Datasets from SemEval2019
    - located at https://zenodo.org/record/1489920?token=eyJhbGciOiJIUzI1NiIsImV4cCI6MTU3NDM3NzE5OSwiaWF0IjoxNTQyODc1MDg5fQ.eyJkYXRhIjp7InJlY2lkIjoxNDg5OTIwfSwiaWQiOjE0MjEsInJuZCI6IjMxNDM1ZGQ3In0.9uT-erF9VVgxHsp6x6RAmCIImAwGzXwlMJWoXzJ1-Zk#.XAF9EBNKhE5

Execution Notes:
    - Currently, the program expects the training word index to be located
    - in the same directory as the training data, with the same name as the
    - training data, but with '.json' attached to the end

    - It also expects the embedding matrix to be located at 'data/embeddings/embedding_matrix.npy'
    - I did not write this code to be very portable :(
    - the embedding matrix is a 7GB text .npy file created from the 
    - get_pretrained_word_embeddings() method in hp.py and then saved to a file
    - I am not going to include the embedding_matrix or W2V file because it is
    - easily available online, and I am running out of free google drive space

    - It will also try to save images of graphs and config files to a specific path
    - and will crash if that path doesn't exist
    
Execution:
    - python hp_lstm.py --training_data <PATH> --training_labels <PATH> --validation_data <PATH> --validation_labels <PATH> --test_data <PATH> --test_labels <PATH>