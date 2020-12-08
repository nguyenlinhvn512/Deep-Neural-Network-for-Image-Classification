# GRADED FUNCTION: compute_cost

def compute_cost(AL, Y):
    # """
    # Implement the cost function defined by equation (7).

    # Arguments:
    # AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    # Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    # Returns:
    # cost -- cross-entropy cost
    # """

    m = Y.shape[1]

    # Compute loss from aL and y.
    ### START CODE HERE ### (≈ 1 lines of code)
    cost = - 1/m * (np.dot(Y, (np.log(AL)).T) + np.dot(1-Y, (np.log(1-AL)).T))
    ### END CODE HERE ###

    # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return cost
