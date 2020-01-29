data {
  real<lower = 0> mean_sd;
  real<lower = 0> rnd_sd;
  int<lower = 0> N;
  vector<lower = 0>[N] s;
  vector[N] eff;
}
parameters {
  real eff_mean;
  real<lower = 0> eff_sd;
  vector[N] ran_eff;
}
transformed parameters {
  vector[N] eff_model = eff_mean + eff_sd * ran_eff;
}
model {
  eff_mean ~ normal(0, mean_sd);
  eff_sd ~ normal(0, rnd_sd);
  ran_eff ~ std_normal();
  eff ~ normal(eff_model, s);
}
