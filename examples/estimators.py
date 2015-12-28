import numpy as np
import scipy as sp
from scipy import stats


def estimate_key(data, algorithm, distance, key_profile_id, normalize=True):
    """Estimates the global key by using the KeyEstimator class.
    For detailed information about the algorithms implemented, please check
    the KeyEstimator class

    Parameters
    ----------
    data : np.ndarray
        Pitch class histogram or pitch class transition matrix, according to the
        algorithm chosen. 'krumhansl_schmucker' uses a pitch class histogram,
        'madsen_widmer' uses a pitch class transition matrix and 'temperley'
        uses beat-aligned chromagrams
    algorithm : str
        String identifier that represents the implemented algorithms.
        ('krumhansl_schmucker', 'madsen_widmer', 'temperley')
    key_profile_id : str
        Represents the key profile ID to instantiate the KeyProfile or
        KeyProfileInterval object. For detailed information, please check the
        KeyProfile or KeyProfileInterval class.
    normalize : boolean
        Normalize KeyProfile data

    Returns
    -------
    results : tuple(float)
        Best key, best score, confidence, (predictions)
        Temperley's algorithm also returns predictions per frame
    """

    if algorithm not in KeyEstimator._algorithms:
        raise Exception('Algorithm {} is not supported'.format(algorithm))

    if algorithm == 'krumhansl_schmucker':
        key_profile = KeyProfile(key_profile_id, normalize)
    elif algorithm == 'madsen_widmer':
        key_profile = KeyProfileInterval(key_profile_id, normalize)
    elif algorithm == 'temperley':
        key_profile = KeyProfile(key_profile_id, normalize)
    else:
        return None

    key_estimator = KeyEstimator(algorithm, distance, key_profile)
    return key_estimator.estimate(data)


class KeyEstimator(object):
    _algorithms = ('krumhansl_schmucker', 'madsen_widmer', 'temperley')

    def __init__(self, algorithm, distance, key_profile):
        """The KeyEstimator class provides key estimation algorithms based on
        the work of Krumhansl and Schmucker, Madsen and Widmer, and Temperley.

        Attributes
        ----------
        algorithm : str
            String representation of the implemented algorithms
            ('krumhansl_schmucker', 'madsen_widmer', 'temperley')
        key_profile : KeyProfile
            KeyProfile or KeyProfileInterval object
        """

        if algorithm not in KeyEstimator._algorithms:
            raise Exception('Algorithm {} is not supported'.format(algorithm))

        # check for valid data
        if algorithm == 'madsen_widmer':
            if type(key_profile) is not KeyProfileInterval:
                raise Exception('Unexpected key_profile {}'.format(key_profile))
        elif algorithm == 'krumhansl_schmucker' or algorithm == 'temperley':
            if type(key_profile) is not KeyProfile:
                raise Exception('Unexpected key_profile {}'.format(key_profile))

        self.algorithm = algorithm
        self.key_profile = key_profile

        # set distance, fit and confidence functions
        self.set_estimator_funcs(algorithm)

    def set_estimator_funcs(self, algorithm):
        if algorithm == 'krumhansl_schmucker':
            # pearson correlation
            self.dist_func = lambda x, y: sp.stats.pearsonr(x, y)[0]
            self.fit_func = lambda x: np.argsort(x)[-2:][::-1]
            self.conf_func = lambda x, y: 0.5 * (1 + ((x - y) / x))
        elif algorithm == 'madsen_widmer':
            # sum of scalar products
            self.dist_func = lambda x, y: np.multiply(x, y).sum()
            self.fit_func = lambda x: np.argsort(x)[-2:][::-1]
            self.conf_func = lambda x, y: 0.5 * (1 + ((x - y) / x))
        elif algorithm == 'temperley':
            # similar to HMM forward algorithm
            def dist_func(chroma, key_profile, zero_indices):
                # probability of existing pitch classes given key
                prob_key = key_profile * chroma
                # probability of non-exisquting pitch classes given key
                prob_key[zero_indices] = 1 - key_profile[zero_indices]
                # probability of key
                prob_key = np.log(prob_key).sum()
                return prob_key

            self.dist_func = dist_func
            self.fit_func = lambda x: np.argsort(x)[-2:][::-1]
            self.conf_func = lambda x, y: (y - x) / y
        else:
            return None

    def estimate(self, data):
        """Estimates the key using the respective algorithm.

        Parameters
        ----------
        data : various
            Data to be used with the respective algorithm.
            'krumhansl_schmucker' expects pitch class histogram
            'madsen_widmer' expects pitch class transition matrix
            'temperley' expects beat aligned chromagrams

        Returns
        -------
        best key, best score, confidence, (predictions) : tuple
            Estimated key using numeric notation, best score and confidence
            Temperley's algorithm also returns predictions per frame
        """

        if self.algorithm == 'krumhansl_schmucker':
            return self.estimate_key_krumhansl_widmer(data)
        elif self.algorithm == 'madsen_widmer':
            return self.estimate_key_madsen_widmer(data)
        elif self.algorithm == 'temperley':
            return self.estimate_key_temperley(data)
        else:
            return None

    def estimate_key_krumhansl_widmer(self, histogram):
        """Estimates the key signature by using Krumhansl and Schmucker
        algorithm.
        Krumhansl, C. "Cognitive Foundations of Musical Pitch, ch. 4"

        Parameters
        ----------
        histogram : np.ndarray, shape=(12,1)
            pitch class histogram
        key_profile : KeyProfile
            KeyProfile object with respective major and minor profiles.

        Returns
        -------
        best_key : int
            Estimated key using numeric notation
        best_scores : float
            Best score given fit function
        confidence : float
            Estimated confidence given scores and confidence function
        """

        # compute distances for major and minor
        dists = np.array(
                [self.dist_func(histogram, np.roll(self.key_profile.major, i))
                    for i in xrange(0, 12)] +
                [self.dist_func(histogram, np.roll(self.key_profile.minor, i))
                    for i in xrange(0, 12)])

        # use fit function to find best matches
        best_keys = self.fit_func(dists)
        best_scores = dists[best_keys]

        # compute confidence using two best scores
        confidence = self.conf_func(best_scores[0], best_scores[1])
        return best_keys[0], best_scores[0], confidence

    def estimate_key_madsen_widmer(self, transition_matrix):
        """Estimates the key signature by using Soren Madsen and Gerhard
        Widmer algorithm.
        Madsen, S. T. and Widmer, G. "Key-Finding With Interval Profiles"

        Parameters
        ----------
        transition_matrix : np.ndarray, shape=(12, 12)
            pitch class transition matrix, can be extracted with
            get_pitch_class_transition_matrix

        Returns
        -------
        best_key : int
            Estimated key using numeric notation
        best_scores : float
            Best score given fit function
        confidence : float
            Estimated confidence given scores and confidence function
        """

        # compute distances for major and minor
        dists = np.array([self.dist_func(
                    transition_matrix, np.roll(self.key_profile.major, i))
                    for i in xrange(0, 12)] +
                [self.dist_func(
                    transition_matrix, np.roll(self.key_profile.minor, i))
                    for i in xrange(0, 12)])

        # use fit function to find best matches
        best_keys = self.fit_func(dists)
        best_scores = dists[best_keys]

        # compute confidence using two best scores
        confidence = self.conf_func(best_scores[0], best_scores[1])
        return best_keys[0], best_scores[0], confidence

    def estimate_key_temperley(self, chromagram, rep_key=.9, hard_mask=True):
        """Estimates the key signature by using Temperley's algorithm.
        Temperley, David. "A Bayesian key-finding model."
        MIREX extended abstract (2005).

        Parameters
        ----------
        chromagram : np.ndarray, shape=(12, ...)
            Chromagram matrix
        rep_key : float
            Harmonic persistence weight for repeated key
        hard_mask : boolean
            Apply hard mask (note present or not) as in Temperley original
            algorithm or soft mask

        Returns
        -------
        best_key : int
            Estimated key using numeric notation
        predictions : np.array,
            Estimated harmonic progression
        """

        # predictions for each chromagram time frame
        predictions = np.zeros(chromagram.shape[1], dtype=int)
        prev_key_number = None
        prev_scores = np.zeros(12)

        # set initial and transition probabilities
        uni_key = np.log(1.0 / 24)
        rep_key = np.log(rep_key)
        dif_key = np.log((1 - rep_key) / 23.0)

        # similar to hmm's forward algorithm
        for frame_idx in xrange(chromagram.shape[1]):
            chroma = chromagram[:, frame_idx]
            # use soft or hard mask
            if hard_mask:
                chroma = chroma.astype(bool)
            else:
                # normalize
                chroma /= chroma.sum()

            # store indices of non-existing pcs
            zero_indices = ~chroma.astype(bool)

            # compute emission scores given temperley's distance function
            scores = np.array([self.dist_func(
                chroma, np.roll(self.key_profile.major, i), zero_indices)
                for i in xrange(0, 12)] + [self.dist_func(
                    chroma, np.roll(self.key_profile.minor, i), zero_indices)
                    for i in xrange(0, 12)])

            # weight scores by transition probabilities
            if frame_idx == 0:
                # uniform prior
                scores += uni_key
            else:
                scores += dif_key
                scores[prev_key_number] += rep_key - dif_key
                scores += prev_scores

            # find highest score
            best_key_number = self.fit_func(scores)[0]
            predictions[frame_idx] = best_key_number

            # update frame_idx and previous key and score
            prev_key_number = best_key_number
            prev_scores = scores

        best_keys = self.fit_func(prev_scores)
        confidence = self.conf_func(prev_scores[best_keys[0]],
                                    prev_scores[best_keys[1]])

        return best_keys[0], prev_scores[best_keys[0]], confidence, predictions


class KeyProfileInterval(object):
    _profile_ids = ('SG', 'bach')

    def __init__(self, profile_id, normalize=True):
        """Create KeyProfileInterval object. Contains the major and minor key
        interval profile

        Attributes
        ----------
        profile_id : str
            String name of the profile to be used, options include:
            'SG' : 'Soren Madsen and Gerhard Widmer'
            'bach'  : 'Extracted from Bach's Two part Inventions'
        normalize : boolean
            Normalize the key profiles such that the sum equals to one
        """

        if profile_id not in KeyProfileInterval._profile_ids:
            raise Exception('Profile {} is not recognized'.format(profile_id))

        if profile_id == 'SG':
            self.major = np.array([15326.0, 2.0, 9356.0, 53.0, 10513.0, 1929.0,
                                   32.0, 6188.0, 9.0, 2949.0, 94.0, 9128.0, 0.0,
                                   66.0, 357.0, 0.0, 54.0, 2.0, 0.0, 6.0, 0.0,
                                   70.0, 0.0, 54.0, 12941.0, 327.0, 9321.0,
                                   104.0, 10435.0, 3363.0, 84.0, 4478.0, 0.0,
                                   1058.0, 57.0, 4249.0, 48.0, 0.0, 133.0, 19.0,
                                   27.0, 22.0, 4.0, 52.0, 2.0, 4.0, 7.0, 1.0,
                                   8527.0, 99.0, 13932.0, 29.0, 7412.0, 9844.0,
                                   171.0, 9408.0, 14.0, 671.0, 9.0, 380.0,
                                   833.0, 0.0, 3938.0, 45.0, 12395.0, 4873.0,
                                   19.0, 5981.0, 42.0, 2819.0, 20.0, 392.0, 0.0,
                                   0.0, 66.0, 0.0, 190.0, 2.0, 123.0, 771.0,
                                   3.0, 177.0, 0.0, 12.0, 9703.0, 0.0, 3373.0,
                                   63.0, 8086.0, 9014.0, 744.0, 16512.0, 32.0,
                                   6439.0, 118.0, 3214.0, 20.0, 0.0, 8.0, 0.0,
                                   16.0, 36.0, 5.0, 29.0, 38.0, 109.0, 8.0, 9.0,
                                   1457.0, 42.0, 843.0, 0.0, 580.0, 2133.0,
                                   136.0, 10278.0, 108.0, 5102.0, 158.0, 3994.0,
                                   68.0, 0.0, 63.0, 4.0, 9.0, 9.0, 0.0, 85.0,
                                   11.0, 224.0, 103.0, 2.0, 8285.0, 72.0,
                                   4934.0, 0.0, 218.0, 103.0, 26.0, 2619.0,
                                   18.0, 5185.0, 3.0, 2657.0])
            self.minor = np.array([8311.0, 121.0, 6263.0, 3395.0, 79.0, 1130.0,
                                   3.0, 2947.0, 195.0, 242.0, 3305.0, 2335.0,
                                   134.0, 9.0, 1.0, 71.0, 0.0, 3.0, 0.0, 0.0,
                                   1.0, 0.0, 4.0, 17.0, 8276.0, 0.0, 5369.0,
                                   7397.0, 70.0, 2791.0, 35.0, 3037.0, 12.0,
                                   234.0, 1552.0, 1187.0, 3537.0, 80.0, 11290.0,
                                   4084.0, 0.0, 4162.0, 1.0, 2108.0, 76.0, 4.0,
                                   328.0, 72.0, 59.0, 0.0, 98.0, 1.0, 45.0,
                                   116.0, 2.0, 48.0, 0.0, 7.0, 0.0, 0.0, 661.0,
                                   9.0, 1994.0, 6017.0, 118.0, 3105.0, 4.0,
                                   4378.0, 418.0, 187.0, 323.0, 7.0, 0.0, 3.0,
                                   4.0, 0.0, 4.0, 2.0, 41.0, 261.0, 3.0, 55.0,
                                   1.0, 7.0, 3219.0, 0.0, 2486.0, 3716.0, 55.0,
                                   5203.0, 175.0, 9816.0, 1063.0, 978.0, 1928.0,
                                   886.0, 116.0, 2.0, 7.0, 46.0, 1.0, 250.0,
                                   17.0, 1859.0, 294.0, 3.0, 362.0, 9.0, 191.0,
                                   3.0, 56.0, 8.0, 1.0, 92.0, 86.0, 1880.0,
                                   13.0, 514.0, 1024.0, 111.0, 1869.0, 6.0,
                                   1335.0, 614.0, 1.0, 261.0, 17.0, 2210.0,
                                   876.0, 1664.0, 2160.0, 2.0, 3131.0, 10.0,
                                   774.0, 43.0, 2.0, 6.0, 0.0, 588.0, 9.0, 94.0,
                                   3.0, 665.0])
            self.major = self.major.reshape((12, 12))
            self.minor = self.minor.reshape((12, 12))
        elif profile_id == 'bach':
            self.major = np.array([29., 0., 62., 2., 25., 3., 5., 28., 0., 36.,
                                   2., 107., 0., 0., 6., 2., 2., 0., 0., 0., 0.,
                                   0., 0., 0., 76., 5., 8., 0., 88., 20., 3.,
                                   24., 1., 6., 0., 30., 0., 2., 0., 0., 12.,
                                   0., 4., 0., 0., 0., 0., 0., 46., 0., 105.,
                                   6., 8., 58., 28., 35., 0., 17., 0., 4., 3.,
                                   0., 22., 0., 79., 2., 2., 46., 0., 11., 1.,
                                   3., 2., 0., 9., 4., 14., 6., 0., 43., 6., 8.,
                                   0., 7., 24., 0., 11., 0., 54., 61., 41., 42.,
                                   0., 54., 3., 27., 0., 0., 0., 0., 1., 1., 1.,
                                   0., 0., 12., 0., 3., 27., 0., 11., 0., 7.,
                                   14., 13., 67., 9., 11., 3., 84., 1., 0., 0.,
                                   0., 0., 0., 0., 1., 0., 15., 0., 6., 91., 3.,
                                   27., 4., 17., 4., 2., 31., 2., 76., 14., 5.])
            self.minor = np.array([21., 1., 73., 12., 1., 12., 2., 13., 6., 9.,
                                   59., 26., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
                                   0., 0., 0., 94., 0., 13., 68., 4., 10., 2.,
                                   22., 0., 0., 14., 3., 19., 0., 94., 8., 0.,
                                   50., 1., 16., 12., 1., 8., 0., 1., 1., 6.,
                                   0., 0., 2., 3., 1., 0., 0., 0., 0., 5., 0.,
                                   11., 83., 1., 15., 0., 57., 5., 3., 13., 5.,
                                   0., 0., 2., 3., 4., 0., 0., 11., 0., 2., 1.,
                                   0., 26., 0., 5., 15., 0., 83., 14., 22., 38.,
                                   19., 7., 3., 3., 0., 3., 2., 0., 14., 0.,
                                   51., 3., 0., 27., 7., 4., 0., 9., 1., 0., 0.,
                                   1., 23., 0., 5., 25., 9., 42., 0., 6., 16.,
                                   3., 9., 0., 10., 37., 31., 9., 0., 20., 0.,
                                   8., 1., 0., 3., 0., 5., 9., 7., 0., 0.])
            self.major = self.major.reshape((12, 12))
            self.minor = self.minor.reshape((12, 12))
        if normalize:
            self.major /= self.major.sum()
            self.minor /= self.minor.sum()


class KeyProfile(object):
    _profile_ids = ('KS', 'KK', 'AE', 'BB', 'TKP')

    def __init__(self, profile_id, normalize=True):
        """Create KeyProfile object. Contains the major and minor key profiles

        Attributes
        ---------
        profile_id : str
            String name of the profile to be used, options include:
            'KS'  : 'Krumhansl and Schmucker'
            'KK'  : 'Krumhansl and Kessler'
            'AE'  : 'Aarden and Essen'
            'BB'  : 'Bellman and Budge'
            'TKP' : 'Temperly and Kostka and Payne'
            http://extras.humdrum.org/man/keycor/
        normalize : boolean
            Normalize the key profiles such that the sum equals to one
        """
        if profile_id not in KeyProfile._profile_ids:
            raise Exception('Profile {} is not recognized'.format(profile_id))

        if profile_id == 'KS':
            self.major = np.array([6.35, 2.33, 3.48, 2.33, 4.38, 4.09,
                                   2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            self.minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                   2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        elif profile_id == 'KK':
            self.major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                   2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            self.minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                   2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        elif profile_id == 'AE':
            self.major = np.array([17.7661, 0.145624, 14.9265, 0.160186,
                                   19.8049, 11.3587, 0.291248, 22.062,
                                   0.145624, 8.15494, 0.232998, 4.95122])
            self.minor = np.array([18.2648, 0.737619, 14.0499, 16.8599,
                                   0.702494, 14.4362, 0.702494, 18.6161,
                                   4.56621, 1.93186, 7.37619, 1.75623])
        elif profile_id == 'BB':
            self.major = np.array([16.80, 0.86, 12.95, 1.41, 13.49, 11.93,
                                   1.25, 20.28, 1.80, 8.04, 0.62, 10.57])
            self.minor = np.array([18.16, 0.69, 12.99, 13.34, 1.07, 11.15,
                                   1.38, 21.07, 7.49, 1.53, 0.92, 10.21])
        else:  # TKP
            self.major = np.array([0.748, 0.060, 0.488, 0.082, 0.670, 0.460,
                                   0.096, 0.715, 0.104, 0.366, 0.057, 0.400])
            self.minor = np.array([0.712, 0.084, 0.474, 0.618, 0.049, 0.460,
                                   0.105, 0.747, 0.404, 0.067, 0.133, 0.330])
        if normalize:
            self.major /= self.major.sum()
            self.minor /= self.minor.sum()


class ChordProfile(object):
    _profile_ids = ('binary', 'weighted')

    def __init__(self, profile_id, normalize=False, boolean=False):
        """Create ChordProfile object. Contains the general chord profiles

        Attributes
        ----------
        profile_id : str
            String name of the profile to be used, options include:
            'binary' : dictionary
                Chord profiles using a one-hot vector
            'weighted' : dictionary
                Chord profile with weighted chord degrees
        normalize : boolean
            Normalize the key profiles such that the sum equals to one
        """

        if profile_id not in ChordProfile._profile_ids:
            raise Exception('Profile {} is not recognized'.format(profile_id))
        elif profile_id == 'binary':
            self.profiles = {
              'maj': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]),
              'min': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
              'aug': np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
              'dim': np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]),
              'maj7': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]),
              'min7': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]),
              '7': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0]),
              'dim7': np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]),
              'hdim7': np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]),
              'minmaj7': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
              'maj6': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]),
              'min6': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]),
              '9': np.array([1, 0,  1, 0, 1, 0, 0, 1, 0, 0, 0, 0]),
              'maj9': np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1]),
              'min9': np.array([1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]),
              'sus4': np.array([1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0])}
        elif profile_id == 'weighted':
            self.profiles = {
              'maj': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]),
              'min': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),
              'aug': np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
              'dim': np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]),
              'maj7': np.array([1, 0, 0, 0, .8, 0, 0, .8, 0, 0, 0, .8]),
              'min7': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]),
              '7': np.array([1, 0, 0, 0, .7, 0, .1, .9, 0, .1, .7, 0]),
              'dim7': np.array([1, 0, 0, .7, 0, 0, .7, 0, 0, .9, 0, 0]),
              'hdim7': np.array([1, 0, 0, .7, 0, 0, .7, 0, 0, 0, .9, 0]),
              'minmaj7': np.array([1, 0, 0, .7, 0, 0, 0, .7, 0, 0, 0, .7]),
              'maj6': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]),
              'min6': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]),
              '9': np.array([1, 0,  1, 0, 1, 0, 0, 1, 0, 0, 0, 0]),
              'maj9': np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1]),
              'min9': np.array([1, 0, .9, .9, 0, 0, 0, .9, 0, 0, .9, 0]),
              'sus4': np.array([1, 0, 0, 0, 0, .9, 0, .9, 0, 0, 0, 0])}
        else:
            return None

        if normalize:
            for name, data in self.profiles.items():
                self.profiles[name] = data / data.sum()


def compute_durational_accent(data, sat_dur=0.75, acc_idx=0.125):
    """Computes the durational accent as described by Parncutt:
        dur_accent = (1 - exp(-dur/sat_dur))^acc_idx
        dur = duration
        sat_dur = saturation duration
        acc_idx = accent index (minimal discernible note duration)

    Parncutt, R. (1994). A perceptual model of pulse salience and metrical
    accent in musical rhythms. Music Perception. 11(4), 409-464.

    Parameters
    ----------
        data : 1d numerical array
            array of note durations in seconds
        sat_dur : float
            saturation duration
        acc_idx : float
            accent index (minimal discernible note duration)

    Returns
    -------
        dur_weights : 1d numerical array
            weights to be applied to the original durations
    """

    dur_accent = lambda dur: np.power(1 - np.exp(-dur/sat_dur), acc_idx)
    dur_weights = map(dur_accent, data)
    return dur_weights
