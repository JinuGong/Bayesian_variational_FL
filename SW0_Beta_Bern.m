iter=3*10^2; % # iteration
experiment=10^3; % # independent experiments
num_agent=10; % # agent
num_loc_data=100; % #local data

%Ground truth
beta_gt=[6;2]; % alpha, beta for synthetic generation

%Graph %network structure
Graph_mat_fully_conn=ones(num_agent)-eye(num_agent);
Graph_mat_central=[zeros(num_agent-1) ones(num_agent-1,1);ones(1,num_agent-1) 0];
Graph_mat_polygon=[zeros(num_agent-1,1) eye(num_agent-1);1 zeros(1,num_agent-1)]+[zeros(1,num_agent-1) 1; eye(num_agent-1) zeros(num_agent-1,1)];

%Dataset generation, generate p with bete-distribution, generate 1 or 0
%based on p (if p>rand(1) => 1)
data_set=betarnd(beta_gt(1),beta_gt(2),num_agent,num_loc_data)>rand(num_agent,num_loc_data);


%Prior
beta_pri=[1;1];


%posterior
beta_save_alpha1=zeros(experiment,iter+1);
beta_save_beta1=zeros(experiment,iter+1);

beta_save_alpha1(:,1)=beta_pri(1)*ones(experiment,1);
beta_save_beta1(:,1)=beta_pri(2)*ones(experiment,1);

graph=Graph_mat_fully_conn;
deg_graph=sum(graph);

for ie=1:experiment
    next_agent=randi(num_agent);
    eta_post=[beta_pri(1)-1;beta_pri(1)+beta_pri(2)-2];
    
    %Approximate likelihood
    eta_likelihood=zeros(2,num_agent);
    
    for i=2:iter+1
        data=data_set(next_agent,:);
        eta_k=eta_likelihood(:,next_agent);
        eta_cavity=eta_post-eta_k;

        eta_post=eta_cavity+[sum(data);num_loc_data];
        beta_save_alpha1(ie,i)=eta_post(1)+1;
        beta_save_beta1(ie,i)=eta_post(2)-eta_post(1)+1;
        eta_likelihood(:,next_agent)=eta_post-eta_cavity;

        next_agent=func_next_agent(next_agent,graph,deg_graph,1);
    end
    
end

%posterior
beta_save_alpha2=zeros(experiment,iter+1);
beta_save_beta2=zeros(experiment,iter+1);

beta_save_alpha2(:,1)=beta_pri(1)*ones(experiment,1);
beta_save_beta2(:,1)=beta_pri(2)*ones(experiment,1);

graph=Graph_mat_central;
deg_graph=sum(graph);

for ie=1:experiment
    
    next_agent=randi(num_agent);
    eta_post=[beta_pri(1)-1;beta_pri(1)+beta_pri(2)-2];
    
    %Approximate likelihood
    eta_likelihood=zeros(2,num_agent);
    
    for i=2:iter+1
        data=data_set(next_agent,:);
        eta_k=eta_likelihood(:,next_agent);
        eta_cavity=eta_post-eta_k;

        eta_post=eta_cavity+[sum(data);num_loc_data];
        beta_save_alpha2(ie,i)=eta_post(1)+1;
        beta_save_beta2(ie,i)=eta_post(2)-eta_post(1)+1;
        eta_likelihood(:,next_agent)=eta_post-eta_cavity;

        next_agent=func_next_agent(next_agent,graph,deg_graph,1);
    end
end

%posterior
beta_save_alpha3=zeros(experiment,iter+1);
beta_save_beta3=zeros(experiment,iter+1);

beta_save_alpha3(:,1)=beta_pri(1)*ones(experiment,1);
beta_save_beta3(:,1)=beta_pri(2)*ones(experiment,1);

graph=Graph_mat_polygon;
deg_graph=sum(graph);

for ie=1:experiment
    
    next_agent=randi(num_agent);
    eta_post=[beta_pri(1)-1;beta_pri(1)+beta_pri(2)-2];
    
    %Approximate likelihood
    eta_likelihood=zeros(2,num_agent);
    
    for i=2:iter+1
        data=data_set(next_agent,:);
        eta_k=eta_likelihood(:,next_agent);
        eta_cavity=eta_post-eta_k;

        eta_post=eta_cavity+[sum(data);num_loc_data];
        beta_save_alpha3(ie,i)=eta_post(1)+1;
        beta_save_beta3(ie,i)=eta_post(2)-eta_post(1)+1;
        eta_likelihood(:,next_agent)=eta_post-eta_cavity;

        next_agent=func_next_agent(next_agent,graph,deg_graph,1);
    end
end


%true posterior
beta_tp=[beta_pri(1)+sum(sum(data_set));beta_pri(2)+num_agent*num_loc_data-sum(sum(data_set))];
% yt=betapdf(x,beta_tp(1),beta_tp(2));

%KL divergence
KL_matrix1=zeros(experiment,iter+1);
KL_matrix2=zeros(experiment,iter+1);
KL_matrix3=zeros(experiment,iter+1);
for ie=1:experiment
    for i=1:iter+1
        
        KL_matrix1(ie,i)=func_KL_beta(beta_save_alpha1(ie,i),beta_save_beta1(ie,i),beta_tp(1),beta_tp(2));
        KL_matrix2(ie,i)=func_KL_beta(beta_save_alpha2(ie,i),beta_save_beta2(ie,i),beta_tp(1),beta_tp(2));
        KL_matrix3(ie,i)=func_KL_beta(beta_save_alpha3(ie,i),beta_save_beta3(ie,i),beta_tp(1),beta_tp(2));

        
    end
end
x=1:1:iter;

y1=mean(KL_matrix1);
y2=mean(KL_matrix2);
y3=mean(KL_matrix3);

%%Credible interval
max_cut=floor(experiment*0.875)+1;
min_cut=floor(experiment*0.125)+1;

y1_75M=zeros(1,iter+1);
y1_75m=zeros(1,iter+1);
for i=1:iter+1
    sort_vec=sort(KL_matrix1(:,i));
    y1_75M(i)=sort_vec(max_cut)-y1(i);
    y1_75m(i)=y1(i)-sort_vec(min_cut);
end

y2_75M=zeros(1,iter+1);
y2_75m=zeros(1,iter+1);
for i=1:iter+1
    sort_vec=sort(KL_matrix2(:,i));
    y2_75M(i)=sort_vec(max_cut)-y2(i);
    y2_75m(i)=y2(i)-sort_vec(min_cut);
end
y3_75M=zeros(1,iter+1);
y3_75m=zeros(1,iter+1);
for i=1:iter+1
    sort_vec=sort(KL_matrix3(:,i));
    y3_75M(i)=sort_vec(max_cut)-y3(i);
    y3_75m(i)=y3(i)-sort_vec(min_cut);
end

loglog(x,y2(x+1),'k-')
hold on
loglog(x,y3(x+1),'r-')
loglog(x,y1(x+1),'b-')

% x_marker=iter/10:iter/10:iter;
x_marker=[1,3,10,30,100,300];

errorbar(x_marker,y2(x_marker+1),y2_75m(x_marker+1),y2_75M(x_marker+1),'ko')
hold on
errorbar(x_marker,y3(x_marker+1),y3_75m(x_marker+1),y3_75M(x_marker+1),'ro')
errorbar(x_marker,y1(x_marker+1),y1_75m(x_marker+1),y1_75M(x_marker+1),'bo')
xlabel('Iteration')
ylabel('KL($q^{(i)}(\theta)\|p(\theta|\mathcal{D})$)','interpreter','latex')
legend('Centralized','Polygon','Fully connected')


grid on
axis([0 iter 0 y1(2)+y1_75M(2)+1])