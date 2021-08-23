clc
clear all
close all

%experimental parameters
iter=5*10^5;
experiment=1*10^1;
num_agent=10;
num_loc_data=10;

%natural gradient parameters
alpha=1;
c=0; %baseline constant
S=3*10^1; %numbr of samples

total_data_num=num_agent*num_loc_data;

%Ground truth
beta_gt=[6;2];

%Graph
Graph_mat_fully_conn=ones(num_agent)-eye(num_agent);
Graph_mat_central=[zeros(num_agent-1) ones(num_agent-1,1);ones(1,num_agent-1) 0];
Graph_mat_polygon=[zeros(num_agent-1,1) eye(num_agent-1);1 zeros(1,num_agent-1)]+[zeros(1,num_agent-1) 1; eye(num_agent-1) zeros(num_agent-1,1)];

graph=Graph_mat_central;
deg_graph=sum(graph);

%Dataset generation, generate p with bete-distribution, generate 1 or 0
%based on p (if p>rand(1) => 1)
data_set=exprnd(betarnd(beta_gt(1),beta_gt(2),num_agent,num_loc_data));

%Prior
beta_pri=[2;2];

%Approximate likelihood
eta_likelihood=ones(2,num_agent);

%posterior
beta_save_alpha1=zeros(experiment,iter+1);
beta_save_beta1=zeros(experiment,iter+1);

beta_save_alpha1(:,1)=beta_pri(1)*ones(experiment,1);
beta_save_beta1(:,1)=beta_pri(2)*ones(experiment,1);

graph=Graph_mat_fully_conn;
deg_graph=sum(graph);

for ie=1:experiment
    next_agent=randi(num_agent);
    eta_post=[beta_pri(1)-1;beta_pri(2)-1];
    
    %Approximate likelihood
    eta_likelihood=zeros(2,num_agent);
    
    for i=2:iter+1
        rho=1*10^(-4); %stepsize
        data=data_set(next_agent,:);
        eta_k=eta_likelihood(:,next_agent);

        sum_grad=0;
        for l=1:num_loc_data
            x=data(l);
            for lt=1:S
                theta=betarnd(eta_post(1)+1,eta_post(2)+1);
                sum_grad=sum_grad+func_nat_grad(eta_post,x,theta,c)/S;
            end
        end
        eta_prev=eta_post;
        eta_post=eta_post-rho*(eta_k-1/alpha*sum_grad);
        if eta_post(1)<-1
            eta_post(1)=-1;
        end
        if eta_post(2)<-1
            eta_post(2)=-1;
        end
        
        
        beta_save_alpha1(ie,i)=eta_post(1)+1;
        beta_save_beta1(ie,i)=eta_post(2)+1;
        
        eta_likelihood(:,next_agent)=eta_post-eta_prev+eta_k;

        next_agent=func_next_agent(next_agent,graph,deg_graph);
    end
end


x=1:1:iter;
x_marker=iter/10:iter/10:iter;


% figure(1)
% plot(x,y1(x+1),'k-')
% errorbar(x_marker,y1(x_marker+1),sig1(x_marker+1),'ko')
% xlabel('Iteration')
% ylabel('KL($q^{(i)}(\theta)\|p_{opt}(\theta)$)','interpreter','latex')
% legend('Centralized')
% grid on

figure(2)
plot(x,beta_save_alpha1(1,x+1),'k-')
hold on
% plot(x,beta_save_alpha1(2,x+1),'k--')
% plot(x,beta_save_alpha1(3,x+1),'k:')
plot(x,beta_save_beta1(1,x+1),'r-')
% plot(x,beta_save_beta1(2,x+1),'r--')
% plot(x,beta_save_beta1(3,x+1),'r:')
% xlabel('Iteration')
% ylabel('KL($q^{(i)}(\theta)\|p_{opt}(\theta)$)','interpreter','latex')

grid on

figure(3)
x_theta=0.001:0.001:1;
theta_gt=func_beta_exp_true_post(data_set,beta_pri,x_theta,total_data_num);
% 
% y1=betapdf(x_theta,beta_save_alpha1(1,10^6),beta_save_beta1(1,10^6));
y2=betapdf(x_theta,beta_save_alpha1(1,5*10^5),beta_save_beta1(1,5*10^5));
y3=betapdf(x_theta,beta_save_alpha1(1,10^5),beta_save_beta1(1,10^5));
y4=betapdf(x_theta,beta_save_alpha1(1,10^4),beta_save_beta1(1,10^4));
y5=betapdf(x_theta,beta_save_alpha1(1,10^3),beta_save_beta1(1,10^3));
y6=betapdf(x_theta,beta_save_alpha1(1,10),beta_save_beta1(1,10));


plot(x_theta,theta_gt,'k-')
hold on
% plot(x_theta,y1,'r-')
plot(x_theta,y2,'r-')
plot(x_theta,y3,'r--')
plot(x_theta,y4,'r-.')
plot(x_theta,y5,'r:')
plot(x_theta,y6,'r:')

