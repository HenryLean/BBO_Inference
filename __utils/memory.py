class Memory(object):
    def __init__(self):
        self.decisions = []
        self.observations = []
        self.estimates = []
        self.var_ests = []
        self.regret = []
        self.performance = []

    def clear(self):
        self.decisions.clear()
        self.observations.clear()
        self.estimates.clear()
        self.var_ests.clear()
        self.regret.clear()
        self.performance.clear()
