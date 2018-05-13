# TRMF
This is a C++ implementation of the basic Temporal Regularized Matrix Factorization (TRMF) algorithm proposed by Hsiang-Fu Yu, Nikhil Rao and Inderjit S. Dhillon in  [their 2016 NIPS paper](https://papers.nips.cc/paper/6160-temporal-regularized-matrix-factorization-for-high-dimensional-time-series-prediction). The implementation uses the Eigen 3 library for dense andsparse linear algebra routines.

This program was created by Daniil Gulevskiy and Fedor Indukaev as aa course project for the Large Scale Machine Learning course at [Yandex school of data analysis](https://yandexdataschool.com/).

## About the TRMF algorithm
The TRMF algorihm seeks to solve the problem of multiple time series forecasting, as this is known to be a hard problem for conventional forecasting algorithms such as ARIMA and DLM. Simple versions of these algorithms only work with each time series separately, thus disregarding the correlations among them. Multidimensional DLM model does exist, but scales very poorly with the dimension. Also these conventional models cannot handle missing values, so heuristics should be used to address missing data.

Suppose we have `n` time series spanned over the same time frame of T time ticks, all organized into a matrix `Y` with `n` rows and `T` columns. The general idea of TRMF is to to find a decomposition `Y ≃ FX` of low rank `k` (where `F` is `n x k` and `X` is `k x T`) and at the same time to infer a time series forecasting model on the matrix `X`. The rows of the matrix `X` can be thought of as the basis time series of the `k`-dimensional latent subspace in the `n`-dimensional space of the original time series.

The forecasting is then done in the `X`-space, and the predicted values are sent back to the `Y`-space with the matrix `F`.

This also naturally allows for missing data, as matrix factorizations are routinely used to the very purpose of missing value imputation, for example in recommender systems.

The 'basic' TRMF algorithm is when the time series model in the X space is just a simple autoregressive model, separate for each row of `X`. The `k` models share the lag set, but have different weights. This allows to avoid the scalability problem, while still accounting for possible correlations in data, as they are captured in the `F`-matrix. 

The decomposition and time series model are found by solving the following optimization problem:

We are given 
* The matrix `Y`
* The mask matrix Omega of the same shape as `Y`, which has zeros where the values of `Y` are missing and ones elsewhere
* a lag set LS, denote `L = max(LS)` 
* set of regularization coefficients `lambda_F`, `lambda_W`, `lambda_X`, `eta`. 

Optimize

![main formula](https://i.imgur.com/OIfOZDE.png)
where 
![another formula](https://i.imgur.com/CE8q6cY.png)

`W` denotes the `k  x |LS|` matrix of autoregression weights.  The norm operator denotes the Frobenius norm.

The optimization is done by alternately optimizing for `X`, `W` and `F`. For each of the three steps different algorithms are used

## Installing

Just clone the repository and run `make`. 


## Usage
`./tmrf -i input_file -o prediction_file -d delimeter -k rank -h horizon -l lags_set -s (if you want to use sparse algorithm) -b (if you want the program to to output the big recovered matrix rather than just the predicted values) -x lambda_x -w lambda_w -f lambda_f -e eta`

* `-i input_file` - location of the input file. The file should be in CSV format with one line corresponding to one time series. Missing values can be marked by any non-numeric sequence of caracters, for example `NaN`, `#nan` or `bob`;
* `-o prediction_file` - where to write the predictions;
* `-d delimeter` - a character separatimg the values in the input file. For example `-d ,`;
* `-k rank` - rank of the factorization;
* `-h horizon` - how many time ticks you want to predict;
* `-l lags_set` - the lags you want to include into the autoregressive model, separated by comma (no spaces in between). For example, `-l 1,2,7`
* `-s` - with this flag, sparse matrices are used in the `X` - step. Optimizing for `X` requires solving a system of linear equations with a `T x T`, so if your `T` (number of time ticks in data) is huge, you should use this flag to avoid memory errors. But if `T` is moderate (say, in the hundreds or first thousands), you're better of using dense matrices, because they work faster, so don't use this flag. Note, that if your lag set is'nt very big, there's probably no point in using very large T, as the temporal model is quite primitive and doesn't need too much data to train.
* `-b` - with this flag, the output matrix include the recovered values for Y, so the shape of output will be `n x (T + horizon)`; without this flag only the predictd values are output and so the shape will be  `n x horizon`;
* `-x lambda_x` - the value for the regularization coefficient `lambda_x`. The corresponding term in the optimization objective penalizes the rows of X disagreeing with the autoregression model. For example, `-x 1000`;
* -`w lambda_w` - the value for the regularization coefficient `lambda_w`. The corresponding term in the optimization objective penalizes large autoregression coefficients. For example, `-w 100`;
* -`f lambda_f` - the value for the regularization coefficient `lambda_f`. The corresponding term in the optimization objective penalizes large coefficients in matrix `F`, which defines the linear mapping from `X`-space  to `Y`-space. For example, `-f 0.01`;
* -`e eta` - the value for the regularization coefficient `eta`. The corresponding term in the optimization objective penalizes large coefficients in matrix `X`. For example, `-e 0.01`;

The resulting matrix is saved in CSV format again, each line corresponding to one time series.

## Getting data

this program comes with a dataset of cryptocurrencies exchanges rate, you can [download it here](https://drive.google.com/file/d/1qDxtI_sPWtIwhuJq92_1-3cv7ecJmsD0/view?usp=sharing).

The files are in a rather peculiar format, you can `converter.py` script, included in this repo, to convert these files to CSV and combine a set of columns from different files into a single file. The script is written in Python 3 and requies Pandas.

Usage:
`converter.py -i file_with_list -o output_file_name -b begin_datetime -e end_datetime -c columns_list`

* `-i file_with_list` - path to a file containing paths to the files that you want to include un the table. An example is of such file is included, it will work if `data` directory is in the same directory that yor executive file is.
* `-b begin_datetime` the time moment you want to start your dataset to start with. If cone cof the files doesnt cover this moment it will be excluded from the result and a corresponding message will be printed. THe datetime should be in the format
 `yyyy.mm.dd.hh.mm.ss`. 
* `-e end_datetime`. Same format as above, if the moment isnt covered, the file will be excluded.
* `-c columns_list`. Column names that you want to include, separated by spaces. For example, `O C` .

Example:
`-i file_list -o converted.csv -b 2016.12.29.00.00.00 -e 2018.02.01.00.00.00 -c O C`







