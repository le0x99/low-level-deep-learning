from math import exp, log


def sigmoid(logits):
    if type(logits) == float:
        return 1 / (1 + exp(-logits))
    else:
        return [sigmoid(logit) for logit in logits]  
def softmax(logits):
    assert type(logits) == list
    logits = logits[0] if type(logits[0]) == list else logits
    sel = sum([exp(logit) for logit in logits])
    return [exp(logit) / sel for logit in logits]

def cross_entropy(y, t):
    y = y[0] if type(y[0]) == list else y
    t = t[0] if type(t[0]) == list else t
    return - sum([t[_] * log(y[_]) for _ in range(len(y))])