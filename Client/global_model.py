from DeepLearning.PPO import MaskablePPO

global_models = {
    # "Reward_win_10M": MaskablePPO.load("DeepLearning/Thesis/Rewards/Models/Reward_win/Reward_win_10M.zip"),
    # "SP_Distribution": MaskablePPO.load("DeepLearning/Thesis/Opponents/Models/Distribution/model_14966784.zip"),
    # "SP_Uniform": MaskablePPO.load("DeepLearning/Thesis/Opponents/Models/Uniform/model_14667776.zip"),
    # "VsModel": MaskablePPO.load("DeepLearning/Thesis/Opponents/Models/VsModel/model_1536000.zip"),
    "VsBaseline": MaskablePPO.load("DeepLearning/Thesis/5.Opponents/Models/DecreasingTurnLimit/model_7954432_5.zip"),
    "SelfPlayDense": MaskablePPO.load("DeepLearning/Thesis/6.DenseRewards/Models/SelfPlayDense/model_19701760_125.zip"),
    "SetupSettlement": MaskablePPO.load("DeepLearning/Thesis/Setup/Models/SetupAgentFirstSettlement/model_716400_1.zip"),
    "SetupCity": MaskablePPO.load("DeepLearning/Thesis/Setup/Models/SetupAgentCities/model_501200_2.zip")
}
