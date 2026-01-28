# config.py
import os

class Config:
    # ================= 路径设置 =================
    # 数据集根目录
    ROOT_DIR = "/home/abc/wq/TWO1/renders"
    # MOS 标签文件
    MOS_FILE = "/home/abc/wq/TWO1/mos_advanced.json"
    # 结果保存主目录
    OUTPUT_DIR = "results"
    
    # ================= 实验名称与描述 =================
    EXP_NAME = "Exp_v1"  # 实验组名
    DESCRIPTION = "Standard modularized training with SSL and Decoupling"

    # ================= 训练超参数 =================
    SEED = 42             # 随机种子，设置为 None 则随机
    GPU_ID = "0"          # 指定 GPU
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    EPOCHS = 50
    LR = 1e-4             # 学习率
    
    # ================= 损失函数权重 (Ablation Control) =================
    # 通过设置权重为 0.0 来实现消融实验 (Ablation)
    LAMBDA_MSE = 1.0      # 主分数回归损失
    LAMBDA_RANK = 0.1     # 排序损失
    LAMBDA_MI = 0.1       # 解耦损失 (MI Loss) -> 设为0即为 w/o Decoupling
    LAMBDA_SUB = 0.5      # 子任务分数损失 -> 设为0即为 w/o Multi-task
    LAMBDA_SSL = 0.2      # 自监督一致性损失 -> 设为0即为 w/o SSL

    # ================= 模型配置 =================
    NUM_FRAMES = 8        # 每个视频采样的帧数
    USE_SUBSCORES = True  # 是否使用子分数头
    USE_FUSION = True     # 是否使用特征融合模块
    
    # ================= 功能开关 =================
    ENABLE_WANDB = False        # 是否使用 WandB 记录
    WANDB_PROJECT = "OF-NeRF-QA"
    SAVE_PER_VIDEO_RESULT = True # 是否保存每个视频的预测结果JSON

    @classmethod
    def get_output_path(cls):
        """自动生成当前实验的保存路径"""
        path = os.path.join(cls.OUTPUT_DIR, cls.EXP_NAME, f"seed_{cls.SEED}")
        os.makedirs(path, exist_ok=True)
        return path
