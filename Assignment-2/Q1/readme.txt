I implemented a method called test_validation which is used for training and it returns the accuracy and loss.

Used it to get training and testing set validation which then I stored in a list. The minimum value stored in the list corresponds to Best Epoch which has less validation error.

Also, I used L2 regularisation:
optimizer = optim.SGD(list(model.fc.parameters()), lr=0.001, momentum=0.9, weight_decay=0.43)

I have an anaconda env setup so I ran the code directly on Pycharm IDE, there was no explicit command which I used to run the code.

torch.save(model.state_dict(), PATH) is used to save the model.

I believe, the code will run with the following command:
Python3 assignment2_code.py