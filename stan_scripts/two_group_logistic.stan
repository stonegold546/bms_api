data {
  real<lower = 0> sd_m; // 3 / qnorm(.975) is reasonable default
  real<lower = 0> sd_m_diff; // log(2, 3, 10) / qnorm(.975) for different levels of belief
  int pass[2];
  int total[2];
}
parameters {
  real m0;
  real m_diff;
}
transformed parameters {
  vector[2] means_logit;
  means_logit[1] = m0;
  means_logit[2] = m0 + m_diff;
}
model {
  m0 ~ normal(0, sd_m);
  m_diff ~ normal(0, sd_m_diff);

  pass ~ binomial_logit(total, means_logit);
}
generated quantities {
  // vector[2] log_lik;
  // vector[2] yhat;
  real odds_ratio = exp(m_diff);
  vector[2] means_prob = inv_logit(means_logit);
  real prob_ratio = means_prob[2] / means_prob[1];
  real prob_diff = means_prob[2] - means_prob[1];

  // for (i in 1:2) {
  //   log_lik[i] = binomial_logit_lpmf(pass[i] | total[i], means_logit[i]);
  //   yhat[i] = binomial_rng(total[i], means_prob[i]);
  // }
}
