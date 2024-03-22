s_train, losses_val, accuracies_train, accuracies_val = mini_batch_gd_sigmoid(
    X_train, Y_train, y_train, X_val, Y_val, y_val, W, b, n_batch=n_batch, n_epochs=n_epochs, eta=eta)
