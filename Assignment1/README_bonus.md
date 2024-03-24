# Assignment 1 Bonus - DD2424 - One Layer Network

> Tristan PERROT

## Exercise 2 - Optional for Bonus Points

### Improve performance of the network

The goal now is to improve the model by adding some techniques.

At first, I retrieve all the data and I normalize it after that I split it into a training, validation and test set. I also implement a grid search to see which parameters are the best for the model.
This is the best result I obtained:

- With old hyperparameters: `lambda=0.1, n_epochs=40, n_batch=100, eta=.001`
![Cost, loss and Accuracy, All Data](Result_Pics/cost_loss_accuracy_all_data_previous_0.1_40_100_0.001.png)

There we can already see a huge improvement compared to the previous model. The accuracy is now **2%** better than before. This is due to the fact that we have more data to train the model and therefore the model is more accurate. However, the model is a really simple model and I think the improvement would be better with a more complex model. However, we can see that on the accuracy graph, the curves are not smooth, certainly due to hyperparameters, which can be improved with the grid search.

- With best hyperparameters: `lambda=0.1, n_batch= 200, n_epochs= 40, eta= 0.005`
![Cost, loss and Accuracy, All Data, Grid Searched](Result_Pics/cost_loss_accuracy_all_data_grid_searched_0.1_40_200_0.005.png)

Now we can see the benefits of the grid search. Unfortunately, the accuracy is not better than the previous model. The hyperparameters are closed to the ones before. Moreover, the accuracy can may us thinking that the network is overfitting.

After that, I implement the **step decay** and I only used all of the data. I re used the best parameters but with an higher learning rate because it will be decayed. I obtained the following results:

- With `lambda=0.2, n_batch=200, n_epochs=40, eta= 0.01`
![Cost, loss and Accuracy, Step Decay](Result_Pics/cost_loss_accuracy_step_decay.png)

Unfortunately, the step decay is not helping so much. But the accuracy is not oscillating as much as before. The model is not overfitting but we could have a better accuracy by using a more complex model, more adapted to the problem. We see there the limits of a simple model.

## Train network - multiple binary cross-entropy losses

After that, I implement the **multiple binary cross-entropy losses** with **sigmoid** activation function on all the dataset.
$$
\sigma(s) = \frac{1}{1+e^{-s}} \\
l_{multiple\ bce} = -\frac{1}{K}\sum_{i=1}^{K} y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \\
z = Wx + b \\
\hat{y} = \sigma(z) \\
$$
$$
\frac{\partial  l_{multiple\ bce}}{\partial W} = \frac{\partial  l_{bce}}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial z} \frac{\partial z}{\partial W} \\
\frac{\partial  l_{multiple\ bce}}{\partial W} = \left( \frac{\hat{y} - y}{\hat{y}(1-\hat{y})} \right) \left( \hat{y}(1-\hat{y}) \right)x = (\hat{y} - y)x \\
\frac{\partial  l_{multiple\ bce}}{\partial b} = \frac{1}{K} \sum_{i=1}^{K} (\hat{y}_i - y_i)
$$

And this is the result I obtained with the hyperparameters: `n_batch=200, n_epochs=50, eta=.001`:
![Histograms Train](Result_Pics/histogram_train.png)
![Histograms Validation](Result_Pics/histogram_val.png)
![Histograms Test](Result_Pics/histogram_test.png)

Here we have not a huge improvement on the accuracy compared to the previous model. The model is not overfitting but we could have a better accuracy by using a more complex model, more adapted to the problem.
