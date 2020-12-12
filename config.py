class DefaultTrainConfig(object):
    sequence_length = 50
    input_size = 100
    hidden_size = 300
    num_layers = 2
    num_classes = 2
    dropout = 0.5
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.01
    k_fold = 10
    test_size = 0.1

    verbose = True
    verbose_step = 100
