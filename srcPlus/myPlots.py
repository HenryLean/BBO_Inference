from .__init__ import *
from .pfo import get_weights
from scipy.stats import norm
import re


ftypes = ['c1', 'c2', 'b1', 'b2']
methods = ['cs', '4p']
estimators = ['x', 'mu', 'var']
environments = ['Chp', 'Exp']
lines = {'cs': '-.', '4p': '--'}
colors = {'cs': 'lightskyblue', '4p': 'orange'}
labels = {'cs': 'constant', '4p': '4-points'}


L = np.array([
    [2, 0],
    [-1, 2]
])
A = L @ L.T
x = np.array([1, -2])
b = A @ x

def mu_fn(x, c=0): return x @ A @ x / 2 - b @ x + c

def sigmoid(y): return 1 / (1 + np.exp(-y))


optVs = {
    'x': x,
    'c1': {'mu': mu_fn(x, c=20), 'var': (1 + np.linalg.norm(x) * np.cos(np.pi * np.linalg.norm(x)))**2},
    'c2': {'mu': 20 * sigmoid(mu_fn(x)/800 - 2), 'var': (1 + np.linalg.norm(x) * np.cos(np.pi * np.linalg.norm(x)))**2},
    'b1': {'mu': mu_fn(x)/2000 + .3}, 
    'b2': {'mu':2 * sigmoid(mu_fn(x)/600 - 2)}
}


p = 103/7; q = -19/14
rs = np.sqrt((q/2)**2 + (p/3)**3)
x1star = (-q/2 + rs)**(1/3) - (q/2 + rs)**(1/3)
y1star = 3 / 14 * (1 - 2 * x1star)
z1star = 1 - x1star - y1star
# print("The optimal allocation is [{:.4f},{:.4f},{:.4f}]".format(z1star, y1star, x1star))
wStar = np.array([z1star, y1star, x1star])


def csvPath(envs, ftype, method, estimator):
    directory = "./output/A2#{}#{}_{}_{}.csv".format(
        envs, ftype, method, estimator
    )
    return directory

def getDFs(envs, ftype):
    dfs = {method: {} for method in methods}
    try:
        for method in methods:
            for est in estimators:
                dfs[method][est] = pd.read_csv(csvPath(envs, ftype, method, est))
        return dfs
    except:
        print('CSV file \"A2#{}#{}\" fails.'.format(envs, ftype))
        return False

    
def get_x_floats(s): return np.array(list(map(float, re.findall('(-?\d+.\d+)', s))))

def get_centroid(row):
    c = np.mean(row.apply(get_x_floats))
    return pd.Series(c)

def get_norm_x(row, x=optVs["x"]):
    s = row.apply(lambda a: np.linalg.norm(x - get_x_floats(a)))
    return pd.Series(s, index=row.index)


def xTuple(cell): return np.array(cell)
def getCentr(row): return np.mean(row.apply(xTuple))
def getWXY(x):
    w = get_weights(x)
    return np.array([w[2], w[1]])
def getNormW(x):
    diff = getWXY(x) - np.array([x1star, y1star])
    return np.linalg.norm(diff)


def myPlots(df, line, color, label, xlog=True, ylog=False, CI=True, alpha=.95):
    n = df.shape[0]
    x = np.log10(1+np.arange(n)) if xlog else np.arange(n)
    m = np.log10(1+df).mean(axis=1) if ylog else df.mean(axis=1)
    plt.plot(x, m, linestyle=line, label=label)
    if CI:
        l = np.log10(df.quantile((1 - alpha)/2, axis=1)) if ylog else df.quantile((1 - alpha)/2, axis=1)
        u = np.log10(df.quantile((1 + alpha)/2, axis=1)) if ylog else df.quantile((1 + alpha)/2, axis=1)
        plt.fill_between(x, l, u, alpha=.3, color=color)

def plotEuclid(df, dpi=500, xlog=True, alpha=.95):
    plt.figure(dpi=dpi)
    for key in methods:
        myPlots(df[key]['x_se'], lines[key], colors[key], labels[key], xlog, alpha=alpha)
    plt.grid(True)
    plt.xlabel(r'$log_{10}$(#iteration)')
    plt.ylabel("Average SE (over sample paths)")
    plt.legend()
    plt.show()
    plt.close()

def plotCentroidPath(df, x=optVs['x'], dpi=500, alpha=.6):
    plt.figure(dpi=dpi)
    for key in methods:
        plt.scatter(df[key]['centroid'].loc[0,0], df[key]['centroid'].loc[0,1], marker='o', alpha=alpha)
        plt.plot(df[key]['centroid'][0], df[key]['centroid'][1], linestyle=lines[key], color=colors[key], label=labels[key], alpha=alpha)
    plt.scatter(x[0], x[1], color='red', marker='*', label='minimizer')
    plt.grid(True)
    plt.xlabel(r"$\theta_{n,1}$")
    plt.ylabel(r"$\theta_{n,2}$")
    plt.legend()
    plt.show()
    plt.close()


def tPlots(tStats, n=1000, dpi=500, knSigma=False, save=False):
    l = min(-1,min(tStats['cs'].loc[n,].min(), tStats['4p'].loc[n,].min()))
    u = min(tStats['cs'].loc[n,].max(), tStats['4p'].loc[n,].max())
    nx = np.linspace(l,u,500)
    ny = norm.pdf(nx)

    plt.figure(dpi=dpi)
    tStats['cs'].loc[n,].hist(density=True, bins=25, histtype='step', label='constant')
    tStats['4p'].loc[n,].hist(density=True, bins=25, alpha=.6, label='4-points')
    if knSigma:
        try:
            tStats['4p*'].loc[n,].hist(density=True, bins=25, color="olive", histtype='step', label='4-points*')
        except: pass

    plt.plot(nx, ny, linestyle="-.", label="std. normal")
    plt.legend()
    plt.grid(True)
    plt.xlabel("normalized estimates")
    plt.ylabel("frequency/density")
    if save:
        try:
            plt.savefig(save)
        except: pass
    plt.show()
    plt.close()

def plotCI(df, line, color, label, xlog=True, ylog=False, alpha=.95):
    n = df['mean'].shape[0]
    x = np.log10(1+np.arange(n)) if xlog else np.arange(n)
    y = np.log10(1+df['mean']) if ylog else df['mean']
    z = norm.ppf((1 + alpha)/2)
    l, u = y - z*df['std'], y + z*df['std']
    plt.plot(x, y, linestyle=line, label=label)
    plt.fill_between(x, l, u, alpha=.3, color=color)

def plotCIs(df, line, color, label, xlog=True, ylog=False):
    n = df['mean'].shape[0]
    x = np.log10(1+np.arange(n)) if xlog else np.arange(n)
    y = np.log10(1+df['mean']) if ylog else df['mean']
    l, u = df["l"].mean(axis=1), df["u"].mean(axis=1)
    plt.plot(x, y, linestyle=line, label=label)
    plt.fill_between(x, l, u, alpha=.3, color=color)
