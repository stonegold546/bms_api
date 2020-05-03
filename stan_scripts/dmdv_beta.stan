data {
  real<lower = 0> sd_m;
  real<lower = 0> sd_m_diff;
  int<lower = 0> N;
  vector<lower = 0, upper = 1>[N] x;
  vector[N] y;
  real min_val;
  real max_val;
}
transformed data {
  vector<lower = 0, upper = 1>[N] y_prop;

  y_prop = (y - min_val) / (max_val - min_val);
  if (min(y_prop) == 0 || max(y_prop) == 1) {
    y_prop = (y_prop * (N - 1) + 0.5) / N;
  }
}
parameters {
  real m0_lgt;
  real m_diff_lgt;
  real<lower = 0> n0;
  real<lower = 0> n1;
}
transformed parameters {
  real<lower = 0, upper = 1> p0 = inv_logit(m0_lgt);
  real<lower = 0, upper = 1> p1 = inv_logit(m0_lgt + m_diff_lgt);
  real<lower = 0> shape0_alpha = p0 * n0;
  real<lower = 0> shape0_beta = (1 - p0) * n0;
  real<lower = 0> shape1_alpha = p1 * n1;
  real<lower = 0> shape1_beta = (1 - p1) * n1;
}
model {
  m0_lgt ~ normal(0, sd_m);
  m_diff_lgt ~ normal(0, sd_m_diff);
  n0 ~ gamma(2, .1);
  n1 ~ gamma(2, .1);

  {
    vector[N] kappa = rep_vector(n0, N);
    for (i in 1:N) if (x[i] == 1) kappa[i] = n1;
    y_prop ~ beta_proportion(
      inv_logit(m0_lgt + x * m_diff_lgt), kappa);
  }
}
generated quantities {
  real m0 = p0 * (max_val - min_val) + min_val;
  real m1 = p1 * (max_val - min_val) + min_val;
  real m_diff = m1 - m0;
  real <lower = 0> st0 = sqrt(p0 * (1 - p0) / (n0 + 1)) * (max_val - min_val);
  real <lower = 0> st1 = sqrt(p1 * (1 - p1) / (n1 + 1)) * (max_val - min_val);
  real<lower = 0> st_ratio = st1 / st0;
}
