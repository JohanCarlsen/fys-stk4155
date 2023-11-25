import sys 
sys.path.insert(0, '../project1')
sys.path.insert(0, '../project1/props')
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from src import set_size

sns.set_theme()

plt.rcParams.update({
'font.size': 8,
'axes.titlesize': 8,
'axes.labelsize': 8,
'xtick.labelsize': 8,
'ytick.labelsize': 8,
'legend.fontsize': 8,
'savefig.bbox': 'tight',
})

def f(x):
    return x**2

def df(x):
    return 2 * x

n = 101
x = np.linspace(-6, 6, n)
y = f(x)

inds = np.arange(n)
rand_inds = np.random.choice(inds, size=n, replace=False)
batches = np.array_split(rand_inds, 10)
guess = 5
guesses = [guess]
tol = 1e-5

i = 0
while abs(df(guess)) >= tol:
    gradient = df(guess)
    guess -= 0.1 * gradient
    guesses.append(guess)
    i += 1

gd_iters = i
gd_guesses = np.array(guesses)
text = f'Est. min. point after {i}\niterations: {guess:.3e}'
fig, ax = plt.subplots(figsize=set_size())
ax.plot(x, y)
ax.plot(gd_guesses, f(gd_guesses), marker='o', ms=3)
ax.text(0.15, 0.75, text, transform=ax.transAxes)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')

fig.savefig('figures/pdfs/gdexample.pdf')
fig.savefig('figures/gdexample.png')
fig.tight_layout()

plt.show()