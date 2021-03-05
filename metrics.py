import numpy as np

'''
Adapted from nltk.metrics.segmentation https://www.nltk.org/_modules/nltk/metrics/segmentation.html
'''
def pk(ref: np.array, hyp: np.array, k: int = None, boundary: int = 1):
    """
    Compute the Pk metric for a pair of segmentations A segmentation
    is any sequence over a vocabulary of two items (e.g. "0", "1"),
    where the specified boundary value is used to mark the edge of a
    segmentation.

    >>> '%.2f' % pk('0100'*100, '1'*400, 2)
    '0.50'
    >>> '%.2f' % pk('0100'*100, '0'*400, 2)
    '0.50'
    >>> '%.2f' % pk('0100'*100, '0100'*100, 2)
    '0.00'
    """

    if k is None:
        k = int(round(ref.shape[0] / (np.count_nonzero(ref == boundary) * 2.0)))

    err = 0.0
    for i in range(len(ref) - k + 1):
        r = np.count_nonzero(ref[i : i + k] == boundary)
        h = np.count_nonzero(hyp[i : i + k] == boundary)
        # print(r)
        # print(h)
        if r != h:
            err += 1
    # print(ref.shape[0])
    return err / (ref.shape[0] - k + 1.0)

'''
Adapted from nltk.metrics.segmentation https://www.nltk.org/_modules/nltk/metrics/segmentation.html
'''
def windowdiff(ref: np.array, hyp: np.array, k: int = None, boundary: int = 1, weighted: bool = False):
    """
    Compute the windowdiff score for a pair of segmentations.  A
    segmentation is any sequence over a vocabulary of two items
    (e.g. "0", "1"), where the specified boundary value is used to
    mark the edge of a segmentation.

        >>> s1 = "000100000010"
        >>> s2 = "000010000100"
        >>> s3 = "100000010000"
        >>> '%.2f' % windowdiff(s1, s1, 3)
        '0.00'
        >>> '%.2f' % windowdiff(s1, s2, 3)
        '0.30'
        >>> '%.2f' % windowdiff(s2, s3, 3)
        '0.80'
    """
    if k is None:
        k = int(round(ref.shape[0] / (np.count_nonzero(ref == boundary) * 2.0)))

    if ref.shape[0] != hyp.shape[0]:
        raise ValueError("Segmentations have unequal length")
    if k > ref.shape[0]:
        raise ValueError(
            "Window width k should be smaller or equal than segmentation lengths"
        )
    wd = 0.0
    for i in range(ref.shape[0] - k + 1):
        ndiff = abs(np.count_nonzero(ref[i : i + k] == boundary) - np.count_nonzero(hyp[i : i + k] == boundary))
        if weighted:
            wd += ndiff
        else:
            wd += min(1, ndiff)
    return wd / (ref.shape[0] - k + 1.0)