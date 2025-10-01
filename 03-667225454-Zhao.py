import numpy as np
import matplotlib.pyplot as plt
import struct
import os
import time


def load_mnist(path, kind="train"):
    labels_path = os.path.join(path, f"{kind}-labels.idx1-ubyte")
    images_path = os.path.join(path, f"{kind}-images.idx3-ubyte")

    with open(labels_path, "rb") as lbpath:
        magic, n = struct.unpack(">II", lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with open(images_path, "rb") as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(num, rows, cols)

    return images, labels


def load_and_prepare_data():
    path = "./mnist"

    train_images_raw, train_labels = load_mnist(path, kind="train")
    test_images_raw, test_labels = load_mnist(path, kind="t10k")

    train_images = train_images_raw.reshape(train_images_raw.shape[0], -1)
    test_images = test_images_raw.reshape(test_images_raw.shape[0], -1)

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_labels_one_hot = np.zeros((train_labels.shape[0], 10))
    train_labels_one_hot[np.arange(train_labels.shape[0]), train_labels] = 1

    return train_images, train_labels, train_labels_one_hot, test_images, test_labels


def train_pta(X_train, D_train, n_samples, eta, epsilon):
    X = X_train[:n_samples]
    D = D_train[:n_samples]

    W = np.random.rand(10, 784) * 0.1

    epoch = 0
    errors_history = []

    print(f"\nTraining with n={n_samples}, eta={eta}, epsilon={epsilon}...")
    start_time = time.time()

    while True:
        current_errors = 0
        for i in range(n_samples):
            x_i = X[i]

            v = W @ x_i

            prediction = np.argmax(v)
            true_label = np.argmax(D[i])

            if prediction != true_label:
                current_errors += 1

        errors_history.append(current_errors)
        error_rate = current_errors / n_samples
        print(
            f"Epoch: {epoch}, Misclassifications: {current_errors}/{n_samples}, Error Rate: {error_rate:.4f}"
        )

        if epoch > 0 and (errors_history[epoch - 1] / n_samples) <= epsilon:
            print("Error threshold reached")
            break

        if epoch > 200:
            print("Reached 200 epochs, stopping to prevent infinite loop.")
            break

        # Update Weights
        for i in range(n_samples):
            x_i = X[i].reshape(784, 1)
            d_i = D[i].reshape(10, 1)

            v = W @ x_i
            u = (v > 0).astype(float)

            error_vector = d_i - u
            W += eta * (error_vector @ x_i.T)

        epoch += 1

    end_time = time.time()
    print(f"Training took {end_time - start_time:.2f} seconds.")
    return W, errors_history


def test_pta(W, X_test, T_test):
    errors = 0
    num_test_samples = X_test.shape[0]

    for i in range(num_test_samples):
        x_i = X_test[i]
        true_label = T_test[i]

        v = W @ x_i
        prediction = np.argmax(v)

        if prediction != true_label:
            errors += 1

    error_percentage = (errors / num_test_samples) * 100
    print(f"Test Results: {errors} misclassified out of {num_test_samples}.")
    return error_percentage


# Plotting
def plot_errors(errors_history, title):
    plt.figure()
    plt.plot(range(len(errors_history)), errors_history, marker="o", linestyle="-")
    plt.title(title)
    plt.xlabel("Epoch Number")
    plt.ylabel("Number of Misclassifications")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    train_imgs, train_lbls, train_lbls_hot, test_imgs, test_lbls = (
        load_and_prepare_data()
    )


    # (e): Experiment with n=50
    print("\n Experiment: n=50, η=1.0, ε=0")
    W_50, errors_50 = train_pta(
        train_imgs, train_lbls_hot, n_samples=50, eta=1.0, epsilon=0
    )
    plot_errors(errors_50, "Training Errors for n=50")
    test_error_50 = test_pta(W_50, test_imgs, test_lbls)
    print(f"Final Test Error for n=50: {test_error_50:.2f}%")


    # (f): Experiment with n=1000
    print("\n Experiment: n=1000, η=1.0, ε=0")
    W_1000, errors_1000 = train_pta(
        train_imgs, train_lbls_hot, n_samples=1000, eta=1.0, epsilon=0
    )
    plot_errors(errors_1000, "Training Errors for n=1000")
    test_error_1000 = test_pta(W_1000, test_imgs, test_lbls)
    print(f"Final Test Error for n=1000: {test_error_1000:.2f}%")


    # (g): Experiment with n=60000
    print("\n Experiment: n=60000, η=1.0, ε=0")
    W_60k_g, errors_60k_g = train_pta(
        train_imgs, train_lbls_hot, n_samples=60000, eta=1.0, epsilon=0
    )
    plot_errors(errors_60k_g, "Training Errors for n=60000")


    # (h): Experiment with n=60000 and 0.15 epsilon --
    print("\n Experiment: n=60000 with 0.15 epsilon")

    for i in range(3):
        print(f"\n Running Trial {i+1} for (h)")
        W_60k_h, errors_60k_h = train_pta(
            train_imgs, train_lbls_hot, n_samples=60000, eta=1.0, epsilon=0.15
        )
        plot_errors(errors_60k_h, f"Training Errors for n=60000 with 0.15 epsilon - Trial {i+1}")
        test_error_60k_h = test_pta(W_60k_h, test_imgs, test_lbls)
        print(f"Trial {i+1} Final Test Error: {test_error_60k_h:.2f}%")
