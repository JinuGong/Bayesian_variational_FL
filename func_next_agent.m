function [next_agent]=func_next_agent(curr_agent,graph,deg_graph,mode)
%mode: MH rule(1), uniformly random(2), round-robin(3)
next_vector=graph(curr_agent,:);
agent_candidates=find(next_vector);
agent_candidate=agent_candidates(randi(length(agent_candidates)));

if mode==1
    trans_prob=min(1,deg_graph(curr_agent)/deg_graph(agent_candidate));
    if rand(1)>trans_prob
        next_agent=curr_agent;
    else
        next_agent=agent_candidate;
    end
elseif mode==2
    next_agent=agent_candidate;
elseif mode==3
    if curr_agent==length(graph(1,:))
        next_agent=1;
    else
        next_agent=curr_agent+1;
    end
end

end
