from sklearn.metrics import f1_score as _f1_score

def f1_score(y, _y, t=0.5):
    y, _y = y.cpu().detach().numpy(), _y.cpu().detach().numpy(),
    bs = _y.shape[0]
    y, _y = y.reshape(bs, -1), _y.reshape(bs, -1)
    score =  _f1_score(y, _y>t, average="micro")
    return score
