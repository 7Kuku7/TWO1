import subprocess

# 格式: (实验名, 参数列表)
experiments = [
    # 1. Full Model
    ("Struct_Full_Model", []), 
    
    # 2. 移除 Multi-task (物理移除 Head)
    ("Struct_wo_MultiTask", ["--no_multitask"]),
    
    # 3. 移除 Decoupling (物理移除 MI Estimator)
    ("Struct_wo_Decoupling", ["--no_decoupling"]),
    
    # 4. 移除 Fusion (物理移除 Attention 模块，改用 Concat)
    ("Struct_wo_Fusion", ["--no_fusion"]),
    
    # 5. 移除 SSL (数据流移除，系数设0)
    ("Struct_wo_SSL", ["--lambda_ssl", "0.0"]),
    
    # 6. 移除 Rank (仅 Loss 移除，无独立模块)
    ("Struct_wo_Rank", ["--lambda_rank", "0.0"]),
]

GPU_ID = "0"
EPOCHS = "50"

print("🚀 开始结构化消融实验...")
for exp_name, args in experiments:
    print(f"\n>>> Running: {exp_name}")
    cmd = ["python", "main.py", "--exp_name", exp_name, "--epochs", EPOCHS, "--gpu", GPU_ID] + args
    subprocess.run(cmd, check=True)
