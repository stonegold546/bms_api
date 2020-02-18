functions {
  real ald_lpdf(vector y, vector mu, vector scale, real prob) {
    real out = sum(log(prob * (1 - prob) ./ scale));
    {
      int len = rows(y);
      vector[len] z = (y - mu) ./ scale;
      for (i in 1:len) {
        if (y[i] < mu[i]) {
          out += (1 - prob) * z[i];
        } else {
          out -= prob * z[i];
        }
      }
    }
    return(out);
  }
}
data {
  real<lower = 0> sd_m; // 5 is reasonable default
  real<lower = 0> sd_m_diff;
  real<lower = 0> sd_st; // 1 is reasonable default
  real<lower = 0> sd_st_r; // log(3) / 2 is reasonable default
  int<lower = 0> N;
  vector<lower = 0, upper = 1>[N] x;
  vector[N] y;
  real<lower = 0, upper = 1> prob;
}
parameters {
  real m0;
  real m_diff;
  real ln_st0;
  real ln_st_ratio;
}
transformed parameters {
  vector[N] mu = m0 + x * m_diff;
  vector[N] sigma = exp(ln_st0 + x * ln_st_ratio);
}
model {
  m0 ~ cauchy(0, sd_m);
  m_diff ~ normal(0, sd_m_diff);
  ln_st0 ~ student_t(3, 0, sd_st);
  ln_st_ratio ~ normal(0, sd_st_r);

  y ~ ald_lpdf(mu, sigma, prob);
}
generated quantities {
  real m1 = m0 + m_diff;
  real <lower = 0> st0 = exp(ln_st0);
  real <lower = 0> st1 = exp(ln_st0 + ln_st_ratio);
  real<lower = 0> st_ratio = exp(ln_st_ratio);
}
