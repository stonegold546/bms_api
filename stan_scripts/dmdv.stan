data {
  real<lower = 0> sd_m; // 5 is reasonable default
  real<lower = 0> sd_m_diff;
  real<lower = 0> sd_st; // 1 is reasonable default
  real<lower = 0> sd_st_r; // log(3) / 2 is reasonable default
  int<lower = 0, upper = 1> nu_choice;
  int<lower = 0> N;
  vector<lower = 0, upper = 1>[N] x;
  vector[N] y;
}
parameters {
  real m0;
  real m_diff;
  real ln_st0;
  real ln_st_ratio;
  real<lower = 0> nu;
}
model {
  m0 ~ cauchy(0, sd_m);
  m_diff ~ normal(0, sd_m_diff);
  ln_st0 ~ student_t(3, 0, sd_st);
  ln_st_ratio ~ normal(0, sd_st_r);
  if (nu_choice == 0) nu ~ gamma(1, 0.1);
  else if (nu_choice == 1) nu ~ exponential(1.0 / 29);

  {
    vector[N] mu = m0 + x * m_diff;
    vector[N] sigma = exp(ln_st0 + x * ln_st_ratio);
    y ~ student_t(nu + nu_choice, mu, sigma);
  }
}
generated quantities {
  real m1 = m0 + m_diff;
  real <lower = 0> st0 = exp(ln_st0);
  real <lower = 0> st1 = exp(ln_st0 + ln_st_ratio);
  real<lower = 0> st_ratio = exp(ln_st_ratio);
}
