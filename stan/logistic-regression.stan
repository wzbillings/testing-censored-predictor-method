//
// Simple logistic regression model
// For naive and complete case analysis
// Zane
// 2024-07-17
//

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N;
  int<lower=0> p;
  array[N] int<lower=0, upper=1> y;
  matrix[N, p] x;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real alpha;
  vector[p] beta;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  y ~ bernoulli_logit_glm(x, alpha, beta);
}

