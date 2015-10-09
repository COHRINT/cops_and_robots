class Grid(object):
    """short description of Grid

    long description of Grid
    
    Parameters
    ----------
    param : param_type, optional
        param_description

    Attributes
    ----------
    attr : attr_type
        attr_description

    Methods
    ----------
    attr : attr_type
        attr_description

    """

    def __init__(self, bounds, res, prior=None):
        self.bounds = bounds
        self.res = res
        self._discretize()

        self.prob = prior

        self.state_transition_matrix = None

    def measurement_update(self, likelihood, measurement=None):
        """Bayesian update of a prior probability with a sensor likelihood.

        Provide likelihood as either a discretized numpy array or as a softmax
        model with an associated measurement class.

        """
        if type(likelihood) != np.ndarray:
            likelihood = likelihood.probability(class_=measurement_label, 
                                                state=self.pos)

        posterior = likelihood * self.prob
        self.prob /= posterior.sum()

    def dynamics_update(self):
        pass

    def find_MAP(self):
        """formerly 'max_point_by_grid'"""
        pass

    def pdf(self, x=None, dims=None):
        pass

    def plot(self, title=None, alpha=1.0, show_colorbar=False, **kwargs):
        pass

    def plot_setup(self):
        pass

    def plot_remove(self):
        pass

    def entropy(self):
        pass

    def compute_KLD(self, other_prob):
        pass

    def _discretize(self):
        pass

def fleming_prior(self):
    pass
