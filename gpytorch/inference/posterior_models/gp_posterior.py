from gpytorch import ObservationModel

class _GPPosterior(ObservationModel):
    def update_data(self, train_xs, train_y):
        """
        Updates this model's training data internally to use the supplied
        train_xs and train_y.

        This method is not intended to update the parameters of the model, that
        is the job of Inference. Only the buffers storing data should be
        updated here.
        """
        pass

    def forward(self, *inputs, **params):
        """
        Given a set of inputs x, returns the predictive posterior distribution
        for the latent function, p(f*|D, x*), where D is the training data as
        set using update_data.
        """
        pass

    def marginal_log_likelihood(self, output, train_y):
        """
        Returns the log marginal likelihood of the data (for exact inference)
        or some lower bound on it (for variational inference).  Inference uses
        this function to optimize or perform sampling for all parameters of the
        model.
        """
        pass