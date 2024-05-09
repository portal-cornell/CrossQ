import gymnasium

gymnasium.register(
    "HumanoidSpawnedUpCustom",
    "envs.humanoid.humanoid_spawned_up:HumanoidEnvCustom",
)

gymnasium.register(
    "HumanoidStandupCustom",
    "envs.humanoid.humanoid_standup_curriculum:HumanoidStandupCustom",
)
