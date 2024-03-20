grad_W, grad_b = compute_gradients(
    X_train[:, :20], Y_train[:, :20], P, W, b, 0)
grad_W_num, grad_b_num = compute_grads_num(
    X_train[:, :20], Y_train[:, :20], P, W, b, 0, 1e-6)
grad_W_num_slow, grad_b_num_slow = compute_grads_num_slow(
    X_train[:, :20], Y_train[:, :20], P, W, b, 0, 1e-6)