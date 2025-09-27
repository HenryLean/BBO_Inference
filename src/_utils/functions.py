import numpy as np



class BaseFn:
    n_dim = 1

    def _validate_x(self, x):
        x = np.array(x, dtype=float).flatten()

        if x.size < self.n_dim:
            return np.pad(x, (0, self.n_dim - x.size), "constant")
        
        return x[:self.n_dim]



class QuadraticFn(BaseFn):
    def __init__(self, A, b, c=1):
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.c = float(c)
        self.n_dim = min(self.A.shape[0], self.b.shape[0])
        self.A = self.A[:self.n_dim, :self.n_dim]
        self.b = self.b[:self.n_dim]

    def __call__(self, x):
        x = self._validate_x(x)
        return 0.5 * x.T @ self.A @ x - self.b @ x + self.c



class SigmoidQuadrFn(QuadraticFn):
    def __init__(self, A, b, c=-1, *, delta=0.01, scale=1):
    # def __init__(self, A, b, c=-4, *, delta=0.01, scale=1/3):
        super().__init__(A, b, c)
        self.delta = delta
        self.scale = scale

    def __call__(self, x):
        u = 1/(1+ np.exp(- super().__call__(x) * self.scale))
        return u + self.delta * (0.5 - u)



class SineFn(BaseFn):
    def __init__(self, A, b, c=0):
        self.A = A
        self.b = b
        self.c = c
        # self.n_dim = self.b.size

    def __call__(self, x):
        x = self._validate_x(x)
        return self.A * np.sin(self.b * np.linalg.norm(x)) + self.c
    