
# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_cycle" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0" "env.episode_length=20" "seq_reward_model=optimal_transport"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_no_door" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0" "env.episode_length=40" "seq_reward_model=temporal_ot"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_bigger" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0" "env.episode_length=40" "seq_reward_model=temporal_ot"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_cycle" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0" "env.episode_length=40" "seq_reward_model=temporal_ot"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=periodic_easy" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0" "env.episode_length=40" "seq_reward_model=temporal_ot"


# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_easy_longer_ref" "env.temporal_encoding=False" "cost_fn=manhattan" "rl_algo.ent_coef=0" "env.episode_length=20" "seq_reward_model=optimal_transport"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_easy_longer_ref" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0" "env.episode_length=20" "seq_reward_model=optimal_transport"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_no_door" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0" "env.episode_length=20" "seq_reward_model=sparse_reward" 

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_cycle" "env.temporal_encoding=False" "cost_fn=manhattan" "rl_algo.ent_coef=0" "env.episode_length=20" "seq_reward_model=even_distribution" 


# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_cycle" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0" "env.episode_length=20" "seq_reward_model=optimal_transport" "rl_algo.gamma=.99"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_cycle" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=20" "seq_reward_model=dtw" "rl_algo.gamma=.99"


# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_cycle" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0" "env.episode_length=20" "seq_reward_model=even_distribution"  "rl_algo.gamma=.9"

############################### Prob Periodic ###############################
# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_bigger" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=prob" "seed=123" 


# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_cycle" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=coverage" "seq_reward_model.name=coverage" "seed=123" 

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_cycle_enforced" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=coverage" "seed=40" 

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_cycle_enforced" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=prob" "seed=40" 

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_cycle_enforced_longer" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=prob" "seed=123" 

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_cycle_enforced_longer" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=coverage" "seed=123" 



# python rl_with_seq_matching.py "env=minigrid" "env.example_name=periodic_easy" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=prob_ranked"  "seq_reward_model.rank_weighting=.1" "seed=123" 

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=periodic_easy" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=even_distribution"  "seed=123" 


# ############################### Prob cyclical ############################
# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_cycle" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=dtw" "seed=123"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_cycle" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=even_distribution" "seed=40"


python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_cycle_enforced_longer" "env.temporal_encoding=True" "cost_fn=weighted_temp_manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=coverage" "seed=123" "logging.wandb_mode=online" "seq_reward_model.tau=1"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_cycle_enforced_longer" "env.temporal_encoding=True" "cost_fn=weighted_temp_manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=log_prob" "seed=123"


# ############################### Prob, SDTW, DTW, SDTW+ with different seeds ############################
# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_bigger" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=prob" "seed=123"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_bigger" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=prob" "seed=40"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_bigger" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=prob" "seq_reward_model.name=ordered_prob_reward" "seed=123"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_bigger" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=prob" "seq_reward_model.name=ordered_prob_reward" "seed=40"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_bigger" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=dtw" "seed=123"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_bigger" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=dtw" "seed=40"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_bigger" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=soft_dtw_plus" "seed=123"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_bigger" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=soft_dtw_plus" "seed=40"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_bigger" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=even_distribution" "seed=123"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_bigger" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=evnen_distribution" "seed=40"



# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_no_door" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0" "env.episode_length=20" "seq_reward_model=dtw" "seed=1234"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_no_door" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0" "env.episode_length=20" "seq_reward_model=soft_dtw_plus" "seed=1234"

################################ Experiment on all sequence algorithms, with and without timestamp ###################################

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_no_door" "env.temporal_encoding=False" "cost_fn=manhattan" "rl_algo.ent_coef=0" "env.episode_length=20" "seq_reward_model=dtw_plus"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_no_door" "env.temporal_encoding=False" "cost_fn=manhattan" "rl_algo.ent_coef=0" "env.episode_length=20" "seq_reward_model=dtw"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_no_door" "env.temporal_encoding=False" "cost_fn=manhattan" "rl_algo.ent_coef=0" "env.episode_length=20" "seq_reward_model=soft_dtw_plus"


# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_cycle" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=optimal_transport" 

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_bigger" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0.1" "env.episode_length=40" "seq_reward_model=optimal_transport" 

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_no_door" "env.temporal_encoding=False" "cost_fn=manhattan"  "rl_algo.ent_coef=0" "env.episode_length=20" "seq_reward_model=even_distribution" 

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_no_door" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0" "env.episode_length=20" "seq_reward_model=dtw_plus"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_no_door" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0" "env.episode_length=20" "seq_reward_model=dtw"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_no_door" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0" "env.episode_length=20" "seq_reward_model=soft_dtw_plus"

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_no_door" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0" "env.episode_length=20" "seq_reward_model=optimal_transport" 

# python rl_with_seq_matching.py "env=minigrid" "env.example_name=lava_nav_no_door" "env.temporal_encoding=True" "cost_fn=manhattan" "rl_algo.ent_coef=0" "env.episode_length=20" "seq_reward_model=even_distribution" 

####################################



