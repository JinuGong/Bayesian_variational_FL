clc
clear all
close all

%experimental parameters
iter=5*10^4;
iter_retrain=5*10^4;
num_agent=10;
num_loc_data=10;

%experimental parameters unlearning
iter_unlearn=3*10^4;
experiment=1*10^1;
agent_fg=num_agent;

%Learnt approximate likelihood
learnt_eta_likelihood=zeros(2,num_agent);


%natural gradient parameters
alpha=1;
c=0; %baseline constant
S=3*10^1; %numbr of samples
rho=5*10^(-3); %step size

total_data_num=num_agent*num_loc_data;

%Ground truth
beta_gt=[2;6];

%Graph
Graph_mat_fully_conn=ones(num_agent)-eye(num_agent);
Graph_mat_central=[zeros(num_agent-1) ones(num_agent-1,1);ones(1,num_agent-1) 0];
Graph_mat_polygon=[zeros(num_agent-1,1) eye(num_agent-1);1 zeros(1,num_agent-1)]+[zeros(1,num_agent-1) 1; eye(num_agent-1) zeros(num_agent-1,1)];


%Dataset generation, generate p with bete-distribution, generate 1 or 0
%based on p (if p>rand(1) => 1)
% for i=1:num_agent
%     data_set(i,:)=exprnd(betarnd(11-i,i,1,num_loc_data));
% end
% data_set=exprnd(betarnd(beta_gt(1),beta_gt(2),num_agent,num_loc_data));

%Prior
beta_pri=[2;2];

%posterior
beta_save_alpha1_learn=zeros(experiment,iter+1);
beta_save_beta1_learn=zeros(experiment,iter+1);
beta_save_alpha1=zeros(experiment,iter_unlearn+1);
beta_save_beta1=zeros(experiment,iter_unlearn+1);
beta_save_alpha1_retrain=zeros(experiment,iter+1);
beta_save_beta1_retrain=zeros(experiment,iter+1);

beta_save_alpha1_learn(:,1)=beta_pri(1)*ones(experiment,1);
beta_save_beta1_learn(:,1)=beta_pri(2)*ones(experiment,1);
beta_save_alpha1_retrain(:,1)=beta_pri(2)*ones(experiment,1);
beta_save_beta1_retrain(:,1)=beta_pri(2)*ones(experiment,1);



x_theta=0.001:0.001:1;
num_data_set=num_agent*num_loc_data;


%KL divergence between approx. posterior and true posterior
KL_matrix_unlearn=zeros(experiment,iter_unlearn+1);
KL_matrix_retrain=zeros(experiment,iter+1);

KL_div_domiain=1:iter_unlearn/100:iter_unlearn;

for ie=1:experiment
    ie
    num_data_set=num_agent*num_loc_data;
    beta_pri=[2;2];
%     for i=1:num_agent
%         data_set(i,:)=exprnd(betarnd(11-i,i,1,num_loc_data));
    % %     end
    for i=1:num_agent-1
        data_set(i,:)=exprnd(betarnd(2,2,1,num_loc_data));
    end
    data_set(num_agent,:)=exprnd(betarnd(1,10,1,num_loc_data));

    num_loc_update=1;

    next_agent=randi(num_agent);
    eta_post=[beta_pri(1)-1;beta_pri(2)-1];
    
    %Approximate likelihood
    eta_likelihood=zeros(2,num_agent);
    
    for i=2:iter+1
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
%         sum_grad
        eta_prev=eta_post;
        eta_post=eta_post-rho*(eta_k-1/alpha*sum_grad);
        
        beta_save_alpha1_learn(ie,i)=eta_post(1)+1;
        beta_save_beta1_learn(ie,i)=eta_post(2)+1;
        
        eta_likelihood(:,next_agent)=eta_post-eta_prev+eta_k;
        
        graph=ones(num_agent)-eye(num_agent);
        deg_graph=sum(graph);
        if mod(i-1,num_loc_update)==0
            
            next_agent=func_next_agent(next_agent,graph,deg_graph,1);
        end
    end

    beta_save_alpha1(ie,1)=beta_save_alpha1_learn(ie,iter+1);
    beta_save_beta1(ie,1)=beta_save_beta1_learn(ie,iter+1);
    


    value=betapdf(x_theta,beta_pri(1),beta_pri(2));
    for k=1:num_agent
        for kk=1:num_loc_data
            value=value.*((1./x_theta)).*exp(-data_set(k,kk)./x_theta);
        end
    end

    norm_factor =sum(value)/length(x_theta);
    beta_posti=[beta_save_alpha1(ie,1);beta_save_beta1(ie,1)];
    func_KLD_beta_exp(beta_pri,beta_posti,norm_factor,num_data_set,data_set)

    figure(ie+1)

%     plot(x_theta,value/norm_factor,'b-');
%     hold on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%         Unlearning          %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    ie
    beta_pri=[beta_save_alpha1(ie,1);beta_save_beta1(ie,1)]
    
    next_agent=randi(num_agent);
    eta_post=[beta_pri(1)-1;beta_pri(2)-1];
    
    %Learnt approximate likelihood
    eta_likelihood=learnt_eta_likelihood;
    
    for i=1:iter_unlearn
        if next_agent~=agent_fg

            data=data_set(next_agent,:);
            eta_k=eta_likelihood(:,next_agent);
            
            eta_prev=eta_post;
            eta_post=eta_post;
            
            next_agent=func_next_agent(next_agent,graph,deg_graph,1);
        else
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
            eta_post=eta_post-rho*(eta_k+sum_grad);
            next_agent=agent_fg;
        end
        
        beta_save_alpha1(ie,i+1)=eta_post(1)+1;
        beta_save_beta1(ie,i+1)=eta_post(2)+1;
        
        eta_likelihood(:,next_agent)=eta_post-eta_prev+eta_k;

    end
    
    beta_pri_before_learn=[2;2];

    value=betapdf(x_theta,beta_pri_before_learn(1),beta_pri_before_learn(2));
    for k=1:num_agent-1
        for kk=1:num_loc_data
            value=value.*((1./x_theta)).*exp(-data_set(k,kk)./x_theta);
        end
    end

    norm_factor=sum(value)/length(x_theta);

    num_data_set=num_data_set-num_loc_data;
    data_set_removed=data_set(1:num_agent-1,:);
    for i=1:iter_unlearn+1

        beta_posti=[beta_save_alpha1(ie,i);beta_save_beta1(ie,i)];
        KL_matrix_unlearn(ie,i)=func_KLD_beta_exp(beta_pri_before_learn,beta_posti,norm_factor,num_data_set,data_set_removed);
    end

% 
%     plot(x_theta,value/norm_factor,'k-');
%     plot(x_theta,betapdf(x_theta,beta_save_alpha1(ie,1),beta_save_beta1(ie,1)),'b--');
%     plot(x_theta,betapdf(x_theta,beta_save_alpha1(ie,iter_unlearn+1),beta_save_beta1(ie,iter_unlearn+1)),'k--');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%         Retraining          %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ie
    beta_pri=[2;2];
    next_agent=randi(num_agent-1);
    eta_post=[beta_pri(1)-1;beta_pri(2)-1];
    
    %Approximate likelihood
    eta_likelihood=zeros(2,num_agent-1);
    
    for i=2:iter_retrain+1
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
%         sum_grad
        eta_prev=eta_post;
        eta_post=eta_post-rho*(eta_k-1/alpha*sum_grad);
        
        beta_save_alpha1_retrain(ie,i)=eta_post(1)+1;
        beta_save_beta1_retrain(ie,i)=eta_post(2)+1;
        
        eta_likelihood(:,next_agent)=eta_post-eta_prev+eta_k;
        
        graph=ones(num_agent-1)-eye(num_agent-1);
        deg_graph=sum(graph);
        if mod(i-1,num_loc_update)==0
            next_agent=func_next_agent(next_agent,graph,deg_graph,1);
        end
    end

    for i=1:iter_retrain+1
        beta_posti=[beta_save_alpha1_retrain(ie,i);beta_save_beta1_retrain(ie,i)];
        KL_matrix_retrain(ie,i)=func_KLD_beta_exp(beta_pri_before_learn,beta_posti,norm_factor,num_data_set,data_set_removed);
    end
% 
%     plot(x_theta,betapdf(x_theta,beta_save_alpha1_retrain(ie,iter_retrain+1),beta_save_beta1_retrain(ie,iter_retrain+1)),'k:');
% 
%     leg=legend('$p(\theta|\mathcal{D}_1,...,\mathcal{D}_{10})$','$p(\theta|{D}_1,...,\mathcal{D}_{9})$','$q(\theta|\theta|{D}_1,...,\mathcal{D}_{10})$','$q(\theta|\theta|{D}_1,...,\mathcal{D}_{10})$ unlearn $\mathcal{D}_{10}$','$q(\theta|\theta|{D}_1,...,\mathcal{D}_{9})$');
%     
%     set(leg, 'Interpreter','latex');

end

figure(1)

y1=mean(KL_matrix_unlearn);
y2=mean(KL_matrix_retrain);

%%Credible interval
max_cut=floor(experiment*0.875)+1;
min_cut=floor(experiment*0.125)+1;

y1_75M=zeros(1,iter_unlearn+1);
y1_75m=zeros(1,iter_unlearn+1);
for i=1:iter_unlearn+1
    sort_vec=KL_matrix_unlearn(:,i);
    y1_75M(i)=sort_vec(max_cut)-y1(i);
    y1_75m(i)=y1(i)-sort_vec(min_cut);
end

y2_75M=zeros(1,iter_retrain+1);
y2_75m=zeros(1,iter_retrain+1);
for i=1:iter_retrain+1
    sort_vec=KL_matrix_retrain(:,i);
    y2_75M(i)=sort_vec(max_cut)-y2(i);
    y2_75m(i)=y2(i)-sort_vec(min_cut);
end

x1=[1, 3];
x2=[10 30];
x3=[100 300];
x4=[1000 3000 10000 30000];
x=[x1 x2 x3 x4];
xr=[x1 x2 x3 x4];
x_total=0:iter_unlearn;

loglog(x_total,y2(x_total+1),'k-')
hold on
loglog(x_total,y1(x_total+1),'r-')
errorbar(x,y1(x+1),y1_75m(x+1),y1_75M(x+1),'ro')
errorbar(xr,y2(xr+1),y2_75m(xr+1),y2_75M(xr+1),'ko')
xlabel('Iteration')
ylabel('KL($q^{(i)}(\theta)\|p(\theta|\mathcal{D}\setminus\mathcal{D}_k)$)','interpreter','latex')
grid on
legend('Retraining','Unlearning')
