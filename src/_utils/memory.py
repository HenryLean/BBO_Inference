class Memory(object):
    def __init__(self):
        self.decisions = []
        self.estimates = []
        self.var_ests = []
        self.distances = []
        self.performance = []
        self.regret = []
        self.y_bars = []

    def collect(
        self, 
        decision, 
        estimate,
        var_est,
        *,
        distance=1,
        performance=0,
        regret=1,
        y_bar=0,
        **kwargs
    ):
        self.decisions.append(decision)
        self.estimates.append(estimate)
        self.var_ests.append(var_est)
        self.distances.append(distance)
        self.performance.append(performance)
        self.regret.append(regret)
        self.y_bars.append(y_bar)


