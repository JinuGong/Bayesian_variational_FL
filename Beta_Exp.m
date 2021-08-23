clc
clear all
close all

%experimental parameters
iter=2*10^4;
experiment=1*10^2;
num_agent=10;
num_loc_data=10;

%natural gradient parameters
alpha=1;
c=0; %baseline constant
S=3*10^1; %numbr of samples
rho=5*10^(-3); %step size

total_data_num=num_agent*num_loc_data;

%Ground truth
beta_gt=[2;6];

%Graph topology/ Fully-connected, Central (star-topology), Polygon (Ring)
Graph_mat_fully_conn=ones(num_agent)-eye(num_agent);
Graph_mat_central=[zeros(num_agent-1) ones(num_agent-1,1);ones(1,num_agent-1) 0];
Graph_mat_polygon=[zeros(num_agent-1,1) eye(num_agent-1);1 zeros(1,num_agent-1)]+[zeros(1,num_agent-1) 1; eye(num_agent-1) zeros(num_agent-1,1)];

%Prior
beta_pri=[2;2];

%posterior matrix
beta_save_alpha1=zeros(experiment,iter+1);
beta_save_beta1=zeros(experiment,iter+1);
beta_save_alpha2=zeros(experiment,iter+1);
beta_save_beta2=zeros(experiment,iter+1);
beta_save_alpha3=zeros(experiment,iter+1);
beta_save_beta3=zeros(experiment,iter+1);
beta_save_alpha4=zeros(experiment,iter+1);
beta_save_beta4=zeros(experiment,iter+1);
beta_save_alpha5=zeros(experiment,iter+1);
beta_save_beta5=zeros(experiment,iter+1);

beta_save_alpha1(:,1)=beta_pri(1)*ones(experiment,1);
beta_save_beta1(:,1)=beta_pri(2)*ones(experiment,1);
beta_save_alpha2(:,1)=beta_pri(1)*ones(experiment,1);
beta_save_beta2(:,1)=beta_pri(2)*ones(experiment,1);
beta_save_alpha3(:,1)=beta_pri(1)*ones(experiment,1);
beta_save_beta3(:,1)=beta_pri(2)*ones(experiment,1);
beta_save_alpha4(:,1)=beta_pri(1)*ones(experiment,1);
beta_save_beta4(:,1)=beta_pri(2)*ones(experiment,1);
beta_save_alpha5(:,1)=beta_pri(1)*ones(experiment,1);
beta_save_beta5(:,1)=beta_pri(2)*ones(experiment,1);

graph=Graph_mat_fully_conn;
deg_graph=sum(graph);



%KL divergence between approx. posterior and true posterior
KL_matrix_fully=zeros(experiment,iter+1);
KL_matrix_cent=zeros(experiment,iter+1);
KL_matrix_poly=zeros(experiment,iter+1);
KL_matrix_4=zeros(experiment,iter+1);
KL_matrix_5=zeros(experiment,iter+1);
KL_div_domiain=1:iter/100:iter;
num_data_set=num_agent*num_loc_data;


x_theta=0.001:0.001:1;
for ie=1:experiment
    %Dataset generation, generate p with bete-distribution
    for i=1:num_agent
        data_set(i,:)=exprnd(betarnd(11-i,i,1,num_loc_data));
    end
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
        eta_prev=eta_post;
        eta_post=eta_post-rho*(eta_k-1/alpha*sum_grad);
        
        beta_save_alpha1(ie,i)=eta_post(1)+1;
        beta_save_beta1(ie,i)=eta_post(2)+1;
        
        eta_likelihood(:,next_agent)=eta_post-eta_prev+eta_k;
        if mod(i-1,num_loc_update)==0
            next_agent=func_next_agent(next_agent,graph,deg_graph,1);
        end
    end

    num_loc_update=10;

    next_agent=randi(num_agent);
    eta_post=[beta_pri(1)-1;beta_pri(2)-1];
    
    %Approximate likelihood
    eta_likelihood=zeros(2,num_agent);
    
    for i=2:iter+1
%         rho=1*10^(-2); %stepsize
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
        
        beta_save_alpha2(ie,i)=eta_post(1)+1;
        beta_save_beta2(ie,i)=eta_post(2)+1;
        
        eta_likelihood(:,next_agent)=eta_post-eta_prev+eta_k;

        if mod(i-1,num_loc_update)==0
            next_agent=func_next_agent(next_agent,graph,deg_graph,1);
        end
    end


    num_loc_update=20;

    next_agent=randi(num_agent);
    eta_post=[beta_pri(1)-1;beta_pri(2)-1];
    
    %Approximate likelihood
    eta_likelihood=zeros(2,num_agent);
    
    for i=2:iter+1
%         rho=1*10^(-2); %stepsize
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
        
        beta_save_alpha3(ie,i)=eta_post(1)+1;
        beta_save_beta3(ie,i)=eta_post(2)+1;
        
        eta_likelihood(:,next_agent)=eta_post-eta_prev+eta_k;

        if mod(i-1,num_loc_update)==0;
            next_agent=func_next_agent(next_agent,graph,deg_graph,1);
        end
    end


    num_loc_update=50;

    next_agent=randi(num_agent);
    eta_post=[beta_pri(1)-1;beta_pri(2)-1];
    
    %Approximate likelihood
    eta_likelihood=zeros(2,num_agent);
    
    for i=2:iter+1
%         rho=1*10^(-2); %stepsize
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
        
        beta_save_alpha4(ie,i)=eta_post(1)+1;
        beta_save_beta4(ie,i)=eta_post(2)+1;
        
        eta_likelihood(:,next_agent)=eta_post-eta_prev+eta_k;

        if mod(i-1,num_loc_update)==0;
            next_agent=func_next_agent(next_agent,graph,deg_graph,1);
        end
    end


    num_loc_update=100;

    next_agent=randi(num_agent);
    eta_post=[beta_pri(1)-1;beta_pri(2)-1];
    
    %Approximate likelihood
    eta_likelihood=zeros(2,num_agent);
    
    for i=2:iter+1
%         rho=1*10^(-2); %stepsize
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
        
        beta_save_alpha5(ie,i)=eta_post(1)+1;
        beta_save_beta5(ie,i)=eta_post(2)+1;
        
        eta_likelihood(:,next_agent)=eta_post-eta_prev+eta_k;

        if mod(i-1,num_loc_update)==0;
            next_agent=func_next_agent(next_agent,graph,deg_graph,1);
        end
    end



    value=betapdf(x_theta,beta_pri(1),beta_pri(2));
    for k=1:num_agent
        for kk=1:num_loc_data
            value=value.*((1./x_theta)).*exp(-data_set(k,kk)./x_theta);
        end
    end

    norm_factor =sum(value)/length(x_theta);


    for i=[2,3,4,5,6,7,8,9,10,11,31,61, 101, 151, 201:iter/200:iter+1]
   
        beta_posti=[beta_save_alpha1(ie,i);beta_save_beta1(ie,i)];
        KL_matrix_fully(ie,i)=func_KLD_beta_exp(beta_pri,beta_posti,norm_factor,num_data_set,data_set);
    end

    ie
    for i=[2,3,4,5,6,7,8,9,10,11,31,61, 101, 151, 201:iter/200:iter+1]

        beta_posti=[beta_save_alpha2(ie,i);beta_save_beta2(ie,i)];
        KL_matrix_poly(ie,i)=func_KLD_beta_exp(beta_pri,beta_posti,norm_factor,num_data_set,data_set);
    end


    for i=[2,3,4,5,6,7,8,9,10,11,31,61, 101, 151, 201:iter/200:iter+1]
      
        beta_posti=[beta_save_alpha3(ie,i);beta_save_beta3(ie,i)];
        KL_matrix_cent(ie,i)=func_KLD_beta_exp(beta_pri,beta_posti,norm_factor,num_data_set,data_set);
    end


    for i=[2,3,4,5,6,7,8,9,10,11,31,61, 101, 151, 201:iter/200:iter+1]
      
        beta_posti=[beta_save_alpha4(ie,i);beta_save_beta4(ie,i)];
        KL_matrix_4(ie,i)=func_KLD_beta_exp(beta_pri,beta_posti,norm_factor,num_data_set,data_set);
    end


    for i=[2,3,4,5,6,7,8,9,10,11,31,61, 101, 151, 201:iter/200:iter+1]
      
        beta_posti=[beta_save_alpha5(ie,i);beta_save_beta5(ie,i)];
        KL_matrix_5(ie,i)=func_KLD_beta_exp(beta_pri,beta_posti,norm_factor,num_data_set,data_set);
    end
end
y1=mean(KL_matrix_fully);
y2=mean(KL_matrix_poly);
y3=mean(KL_matrix_cent);
y4=mean(KL_matrix_4);
y5=mean(KL_matrix_5);


%%Credible interval
max_cut=floor(experiment*0.875)+1;
min_cut=floor(experiment*0.125)+1;

y1_75M=zeros(1,iter+1);
y1_75m=zeros(1,iter+1);
for i=1:iter+1
    sort_vec=KL_matrix_fully(:,i);
    y1_75M(i)=sort_vec(max_cut)-y1(i);
    y1_75m(i)=y1(i)-sort_vec(min_cut);
end
y2_75M=zeros(1,iter+1);
y2_75m=zeros(1,iter+1);
for i=1:iter+1
    sort_vec=KL_matrix_poly(:,i);
    y2_75M(i)=sort_vec(max_cut)-y2(i);
    y2_75m(i)=y2(i)-sort_vec(min_cut);
end
% 
y3_75M=zeros(1,iter+1);
y3_75m=zeros(1,iter+1);
for i=1:iter+1
    sort_vec=KL_matrix_cent(:,i);
    y3_75M(i)=sort_vec(max_cut)-y3(i);
    y3_75m(i)=y3(i)-sort_vec(min_cut);
end
% 
y4_75M=zeros(1,iter+1);
y4_75m=zeros(1,iter+1);
for i=1:iter+1
    sort_vec=KL_matrix_4(:,i);
    y4_75M(i)=sort_vec(max_cut)-y4(i);
    y4_75m(i)=y4(i)-sort_vec(min_cut);
end
% 
y5_75M=zeros(1,iter+1);
y5_75m=zeros(1,iter+1);
for i=1:iter+1
    sort_vec=KL_matrix_5(:,i);
    y5_75M(i)=sort_vec(max_cut)-y5(i);
    y5_75m(i)=y5(i)-sort_vec(min_cut);
end


figure(1)
x_total=[2,3,4,5,6,7,8,9,10,11,31,61, 101, 151, 201:iter/200:iter+1]-1;

x1=[1, 3];
x2=[10 30];
x3=[100 300];
x4=[1000 3000 10000 20000];
x=[x1 x2 x3 x4];
x_theta=0.001:0.001:1;
loglog(x_total,y5(x_total+1),'k-')
hold on
loglog(x_total,y4(x_total+1),'b-')
loglog(x_total,y3(x_total+1),'r-')
loglog(x_total,y2(x_total+1),'c-')
loglog(x_total,y1(x_total+1),'g-')
errorbar(x,y1(x+1),y1_75m(x+1),y1_75M(x+1),'go')
errorbar(x,y2(x+1),y2_75m(x+1),y2_75M(x+1),'co')
errorbar(x,y3(x+1),y3_75m(x+1),y3_75M(x+1),'ro')
errorbar(x,y4(x+1),y4_75m(x+1),y4_75M(x+1),'bo')
errorbar(x,y5(x+1),y5_75m(x+1),y5_75M(x+1),'ko')

xlabel('Iteration')
ylabel('KL($q^{(i)}(\theta)\|p(\theta|\mathcal{D})$)','interpreter','latex')
leg=legend('$L=100$','$L=50$','$L=20$','$L=10$','$L=1$');
set(leg, 'Interpreter','latex');

grid on

figure(2)

plot(x_theta,value/norm_factor,'k--');
hold on
plot(x_theta,betapdf(x_theta,beta_save_alpha1(iter+1),beta_save_beta1(iter+1)),'r--')
