[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# Feedforward Neural Network in D
A simple implementation of a feed-forward neural network in D.

This is currently a single layer NN for binary classification using a sigmoid layer and simple delta-based training method, whereby the weights update per entry in the dataset, which are looped over for a number of epochs.

## How to use
The executable can run directly, or, alternatively, it can be re-built with DUB and run as follows:
```
git clone git@github.com:patrickm663/feedforward-nn.git
cd feedforward-nn/
dub build
dub run
```

## TODO
- Implement logic to split the data into training and testing based on a user's parameter
- Test on more datasets (importing the "wine.csv" dataset to have something to test was one of the biggest hurdles)
- Construct a `neuralnetwork` class that implements the functions as methods rather, in order to create an 'API'
- Test whether it can be called from other languages
- Add test cases
- Add additional layers, more activation functions, etc.

## License
MIT licensed.

## Contributions
All contributions are welcome. I am very new to D, but really enjoying the language thus far!
