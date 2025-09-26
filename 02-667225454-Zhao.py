import numpy as np
import matplotlib.pyplot as plt


def perceptron_train(X_aug, y, w_init, eta=1, max_epochs=1000):
    w = w_init.copy().astype(float)
    miscls_history = []

    for epoch in range(max_epochs):
        misclassified = 0

        for i in range(len(X_aug)):
            prediction = np.sign(X_aug[i].dot(w))

            if prediction != y[i]:
                w += eta * y[i] * X_aug[i]
                misclassified += 1

        miscls_history.append(misclassified)

        if misclassified == 0:
            break

    return w, miscls_history


def main():
    rng = np.random.default_rng(42)

    # ============================ (a) - (g) ===========================
    # Initialize w0, w1, w2
    w0 = rng.uniform(-1 / 4, 1 / 4)
    w1 = rng.uniform(-1, 1)
    w2 = rng.uniform(-1, 1)
    w = np.array([w0, w1, w2])
    print(f"w0: {w0}, w1: {w1}, w2: {w2}")

    num_points = 100
    S = rng.uniform(-1, 1, (num_points, 2))

    X_aug = np.hstack([np.ones((num_points, 1)), S])

    scores1 = X_aug.dot(w)
    y = np.where(scores1 >= 0, 1, -1)

    S1 = S[y == 1]
    S0 = S[y == -1]

    plt.figure(figsize=(8, 8))
    plt.scatter(S1[:, 0], S1[:, 1], c="b", marker="o", label="S1 (+1)")
    plt.scatter(S0[:, 0], S0[:, 1], c="r", marker="x", label="S0 (-1)")

    xs = np.linspace(-1.1, 1.1, 200)
    ys = -(w0 + w1 * xs) / w2

    plt.plot(xs, ys, "k-", label="Boundary")

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    # ============================ (h) - (i) ===========================
    w_prime = rng.uniform(-1, 1, size=3)
    print("Initial random weights w':", w_prime)

    final_w_eta1, miscls_eta1 = perceptron_train(X_aug, y, w_prime, eta=1)

    print("Final weights (n=100, η=1):", final_w_eta1)
    print("Epochs:", len(miscls_eta1))
    print("Misclassifications per epoch:", [int(x) for x in miscls_eta1])

    """
    (h)-vii:
        Based on the results:
            Initial random weights w': [-0.71220499 -0.97212742 -0.54068794]
            Final weights (η=1): [ 1.28779501 -1.2710318   5.77886983]
            Misclassifications per epoch: [17, 9, 12, 6, 4, 4, 4, 0]
        The optimal weights' w0 increased, w1 decreased, w2 increased.
    """

    plt.figure()
    plt.plot(range(len(miscls_eta1)), miscls_eta1)
    plt.xlabel("Epoch")
    plt.ylabel("Misclassifications")
    plt.title("Misclassifications (n=100, η=1)")
    plt.grid(True)
    plt.show()

    # ============================ (j) - (k) ===========================
    final_w_eta10, miscls_eta10 = perceptron_train(X_aug, y, w_prime, eta=10)
    print("Final weights (n=100, η=10):", final_w_eta10)
    print("Epochs:", len(miscls_eta10))
    print("Misclassifications per epoch:", [int(x) for x in miscls_eta10])

    final_w_eta01, miscls_eta01 = perceptron_train(X_aug, y, w_prime, eta=0.1)
    print("Final weights (n=100, η=0.1):", final_w_eta01)
    print("Epochs:", len(miscls_eta01))
    print("Misclassifications per epoch:", [int(x) for x in miscls_eta01])

    # Plotting η=0.1, η=1, η=10
    plt.figure()
    plt.plot(
        range(len(miscls_eta01)), miscls_eta01, label="η=0.1", marker="o", markersize=5
    )
    plt.plot(
        range(len(miscls_eta1)), miscls_eta1, label="η=1", marker="^", markersize=5
    )
    plt.plot(
        range(len(miscls_eta10)), miscls_eta10, label="η=10", marker="s", markersize=5
    )

    plt.xlabel("Epoch")
    plt.ylabel("Misclassifications")
    plt.title("Misclassifications (n=100, η=0.1, η=1, η=10)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ============================ (l) - (m) ===========================
    """
    (l):
        A larger learning rate means faster movement, which can lead to faster convergence in some cases, but it may also exceed the optimal solution, causing constant oscillation before reaching the answer.
        A smaller learning rate can avoid over-adjustment, but if the initial weights are far from the solution, a smaller learning rate may require more epochs.

    (m):
        This won't yield the same results. 
        First, if the starting weights are close to the valid solution, convergence will take fewer epochs. Otherwise, more epochs will be required.
        Second, if the random data S is simple and has clear boundaries, convergence will be fast. Otherwise, convergence will be slow.
        Finally, each change in weight creates a new classification problem for the boundary line w0+w1x1+w2x2 = 0 due to changes in the intercept and slope of the line. The perceptron needs to learn a completely different boundary.
    """

    # ============================ (n) ===========================
    n2 = 1000
    X2 = rng.uniform(-1, 1, size=(n2, 2))
    X2_aug = np.hstack([np.ones((n2, 1)), X2])
    scores2 = X2_aug.dot(w)
    y2 = np.where(scores2 >= 0, 1, -1)

    final_w_eta1_n2, miscls_eta1_n2 = perceptron_train(X2_aug, y2, w_prime, eta=1)
    print("Final weights (n=1000, η=1):", final_w_eta1_n2)
    print("Misclassifications per epoch:", [int(x) for x in miscls_eta1_n2])

    final_w_eta10_n2, miscls_eta10_n2 = perceptron_train(X2_aug, y2, w_prime, eta=10)
    print("Final weights (n=1000, η=10):", final_w_eta10_n2)
    print("Misclassifications per epoch:", [int(x) for x in miscls_eta10_n2])

    final_w_eta01_n2, miscls_eta01_n2 = perceptron_train(X2_aug, y2, w_prime, eta=0.1)
    print("Final weights (n=1000, η=0.1):", final_w_eta01_n2)
    print("Misclassifications per epoch:", [int(x) for x in miscls_eta01_n2])

    print(f"n=1000, η=1 epochs: {len(miscls_eta1_n2)}")
    print(f"n=1000, η=10 epochs: {len(miscls_eta10_n2)}")
    print(f"n=1000, η=0.1 epochs: {len(miscls_eta01_n2)}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(miscls_eta1_n2)), miscls_eta1_n2, label="η=1")
    plt.plot(range(len(miscls_eta10_n2)), miscls_eta10_n2, label="η=10")
    plt.plot(range(len(miscls_eta01_n2)), miscls_eta01_n2, label="η=0.1")
    plt.xlabel("Epoch")
    plt.ylabel("Misclassifications")
    plt.title("Misclassifications (n=1000)")
    plt.legend()
    plt.grid(True)
    plt.show()

    """
    (n):
        First, as the dataset gets larger, the number of epochs also increases, requiring the algorithm to adapt to these points.
        Second, satisfying this condition for all 1,000 points is much stricter than satisfying it for only 100 points, forcing the algorithm to run more rounds until every point is correctly classified.
    """


if __name__ == "__main__":
    main()
