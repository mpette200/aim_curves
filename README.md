### Curve Fitting Share Prices on the AIM Market
Share prices often grow exponentially. Not a very ambitious project to try and fit an exponential growth curve to share prices. Code written in python using scipy library.

The curves are fitted to this equation:
```
y = A*t + B*e^(C*t) + D
```

The shares that most closely fit this curve are selected and plotted by the python code.
