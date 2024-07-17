//
// Logisti regressionc model with a single censored predictor
// Censoring handled by constrained imputation
// Zane Billings
// 2024-07-15
//

// The input data consists of:
// N: positive scalar integer. The number of data records.
// y: real array 0/1 of length N. The vector of outcome variable observations.
// x: real vector of length N. The vector of predictor variable observations.
// c: integer array of length N, either 0 or 1 for all entries.
//   c[i] = 0 if observation i is observed and c[i] = 1 if observation i is
//   censored.
// u: real vector of length N. The vector of lower limits of detection.
data {
  int<lower=0> N;
  int<lower=0> N_obs;
  array[N] int<lower = 0, upper = 1> y;
  array[N_obs] real x_obs;
  real LoD;
}

// Transformed data
// Computed from passed data
transformed data {
  // Number of censored and observed observations
  int<lower=0> N_cens = N - N_obs;
}

// The parameters accepted by the model.
parameters {
  // Regression parameters
  real a;
  real b;
  
  // x distribution parameters
  real x_mu;
  real <lower=0> x_sd;
  
  // Vector of censored x values
  array[N_cens] real<upper=LoD> x_cens;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  // x holder
  vector[N] x;
  
  // Priors
  a ~ normal(0, 1);
  b ~ normal(0, 1);
  x_mu ~ normal(0, 2);
  x_sd ~ normal(0, 2);
  
  // Likelihood
  x_obs ~ normal(x_mu, x_sd);
  x_cens ~ normal(x_mu, x_sd);
  x = to_vector(append_array(x_obs, x_cens));
  y ~ bernoulli_logit(a + b * x);
}

