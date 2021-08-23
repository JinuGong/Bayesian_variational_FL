function [divergence]=func_KL_beta(alpha1,beta1,alpha2,beta2)
%KL(beta(alpha1,beta1)|beta(alpha2,beta2))
alpha_part=(alpha2-alpha1)*psi(alpha1);
beta_part=(beta2-beta1)*psi(beta1);
mix_part=(alpha2-alpha1+beta2-beta1)*psi(alpha1+beta1);

divergence=log(beta(alpha2,beta2)/beta(alpha1,beta1))-alpha_part-beta_part+mix_part;
end