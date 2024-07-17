//
// Regression model with a single censored predictor
// Censoring handled by constrained imputation
// Zane Billings
// 2024-07-15
//

// The input data consists of:
// N: positive scalar integer. The number of data records.
// y: real vector of length N. The vector of outcome variable observations.
// x: real vector of length N. The vector of predictor variable observations.
// c: integer array of length N, either 0 or 1 for all entries.
//   c[i] = 0 if observation i is observed and c[i] = 1 if observation i is
//   censored.
// u: real vector of length N. The vector of lower limits of detection.
data {
  int<lower=0> N;
  vector[N] y;
  vector[N] x;
}

// The parameters accepted by the model.
parameters {
  // Regression parameters
  real a, b;
  real<lower=0> sigma;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  // Priors
  a ~ normal(0, 2);
  b ~ normal(0, 2);
  sigma ~ normal(0, 2);
  
  // Likelihood
  y ~ normal(a + b * x, sigma);
}

