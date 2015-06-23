
class MultimodalSoftmax(Softmax):
    """short description of MultimodalSoftmax

    long description of MultimodalSoftmax
    
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

    def __init__(self):
        super(MultimodalSoftmax, self).__init__()
        self.param = param
        

    def _combine_mms(self):
        """Combine classes with the same label.

        """
        # Convert classes to subclasses
        self.subclass_probs = self.probs[:]
        self.subclass_labels = self.class_labels[:]
        self.subclass_colors = self.class_colors[:]
        self.subclass_cmaps = self.class_cmaps[:]
        self.subclass_weights = self.weights[:]
        self.subclass_biases = self.biases[:]

        self.num_subclasses = self.subclass_probs.shape[1]
        self.num_classes = len(set(self.class_labels))
        self.class_labels = []
        [self.class_labels.append(l) for l in self.subclass_labels
         if l not in self.class_labels]

        # Assign new colors to unique classes
        self.class_cmaps = []
        self.class_colors = []
        for i, label in enumerate(self.class_labels):
            self.class_cmaps.append(
                next(self.subclass_cmaps[j]
                    for j, slabel in enumerate(self.subclass_labels)
                    if label == slabel))
            self.class_colors.append(
                next(self.subclass_colors[j]
                    for j, slabel in enumerate(self.subclass_labels)
                    if label == slabel))

        # Merge probabilities from subclasses
        j = 0
        h = 0
        remaining_labels = self.subclass_labels[:]
        old_label = ''  # <>TODO: get rid of this. TLC would find it unpretty.
        remaining_probs = self.subclass_probs[:]
        self.probs = np.zeros((self.subclass_probs.shape[0], self.num_classes))
        for label in remaining_labels:
            indices = [k for k, other_label in enumerate(remaining_labels)
                       if label == other_label]
            probs = remaining_probs[:, indices]
            self.probs[:, h] += np.sum(probs, axis=1)

            remaining_labels = [remaining_labels[k]
                                for k, _ in enumerate(remaining_labels)
                                if k not in indices]
            remaining_probs = np.delete(remaining_probs, indices, axis=1)
            if len(remaining_labels) == 0:
                break
            else:
                j += 1
                if label != old_label:
                    h += 1
                old_label = label


        def probs_at_state(self, state, class_=None,):
        """Find the probabilities for each class for a given state.

        Parameters
        ----------
        state : array_like
            A a 1-dimensional array containing the state values of all `N`
            states at which to find the probabilities of each class
            specified.
        class_ : int, optional
            The zero-indexed ID of the class.

            Defaults to `None`, which will provide all classes' probabilities.

        Returns
        -------
        An array_like object containing the probabilities of the class(es)
        specified.

        """

        # <>TODO: Subtract a constant from all weights to prevent overflow:
        # http://ufldl.stanford.edu/wiki/index.php/Exercise:Softmax_Regression

        # <>TODO: Refactor for MMS class

        if type(class_) is str:
            class_ = self.class_labels.index(class_)

        if hasattr(self, 'subclass_weights'):
            weights = self.subclass_weights[:]

            # Define the softmax normalizer
            sum_ = np.zeros(weights.shape[0])
            for i, weight in enumerate(weights):
                exp_term = np.dot(state, weight) + self.biases[i]
                sum_[i] = np.exp(exp_term)
            normalizer = sum(sum_)  # scalar value

            # Define each class' probability
            probs = np.zeros(self.num_classes)
            for i in range(self.num_classes):
                exp_term = 0

                #Use all related subclasses
                for j in range(self.num_subclasses):
                    if self.class_labels[i] == self.subclass_labels[j]:
                        exp_term += np.exp(np.dot(state, self.subclass_weights[j, :])\
                            + self.subclass_biases[j])
                probs[i] = exp_term / normalizer
        else: