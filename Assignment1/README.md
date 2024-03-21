# Assignment 1 - DD2424 - One Layer Network

> Tristan PERROT

## Exercise 1

After pre-processing our data, we were asked to implement function to prepare the training of a one layer neural network. After implementing this function in the file `Assignment1.py`. We were asked to test with given gradient checking function.

With a **mini batch** size of 20 and a **lambda** of 0, I obtained the following results:

```output
Relative error grad_W: 0.007670216306639771
Relative error grad_b: 6.913182419900119e-06
Relative error grad_W_slow: 4.8046925072640953e-05
Relative error grad_b_slow: 1.0773036776280925e-08
```

Also with **dimension** reduced to 2 and the same parameters, I obtained the following results:

```output
Relative error grad_W: 6.523845438776908e-06
Relative error grad_b: 0.0008241041050076656
Relative error grad_W_slow: 5.8520390237122694e-08
Relative error grad_b_slow: 3.96584525426006e-06
```

The relative error is quite low and the two first lines is bigger because the gradient computation is not the most accurate, it uses the *finite difference method*. The two last lines are the most accurate because they use the *centered difference formula*.
