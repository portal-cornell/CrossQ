import gymnasium

gymnasium.register(
    "HumanoidSpawnedUp",
    "envs.humanoid.humanoid_spawned_up:VLMRewardedHumanoidEnv",
)

gymnasium.register(
    "HumanoidStandupCurriculum",
    "envs.humanoid.humanoid_standup_curriculum:HumanoidStandupCurriculum",
)
