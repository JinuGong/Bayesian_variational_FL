function [nat_grad]=func_nat_grad(eta,x,theta,c)
eta1=eta(1);
eta2=eta(2);
val2=-psi(1,eta1+eta2+2);
val1=psi(1,eta1+1)+val2;
val4=psi(1,eta2+1)+val2;

inv_fim=1/(val1*val4-val2^2)*[val4, -val2;
    -val2, val1];
grad_mat=[(psi(eta1+eta2+2)-psi(eta1+1))+log(theta);
    (psi(eta1+eta2+2)-psi(eta2+1))+log(1-theta)];
nat_grad=(-log(theta)-x/theta-c)*inv_fim*grad_mat;

end
