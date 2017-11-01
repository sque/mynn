
from scipy.special import expit
from .optimizers import AdaptiveGradientDescentMomentum

SMALL_FLOAT = np.float(1.0e-16)

def my_sigmoid(x):
   return 1 / (1 + np.exp(-x))

def cross_entropy_loss(A:np.ndarray, Y:np.ndarray):
    logprobs = np.multiply(np.log(A + SMALL_FLOAT), Y) + np.multiply((1 - Y), np.log((1 - A) + SMALL_FLOAT))
    return np.squeeze(- np.mean(logprobs))

