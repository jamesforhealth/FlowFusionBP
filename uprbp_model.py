# main_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from uprbp_model import UPRBPEncoder, ProjectionHead, vicreg_loss, tnc_loss, BPRegressor

#####################################
# 自定義 Aurora-BP Dataset
#####################################
class AuroraBPDataset(Dataset):
    def __init__(self, csv_file, unlabeled=False, transform=None):
        """
        :param csv_file: CSV 檔案路徑，必須含有 'ppg_signal' 欄位，
                         標記資料另外必須有 'sbp' 與 'dbp'
        :param unlabeled: 若為 True 則不返回 BP 標記
        """
        self.data = pd.read_csv(csv_file)
        self.unlabeled = unlabeled
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # 假設 'ppg_signal' 為以逗號分隔的數字字串
        ppg_str = row['ppg_signal']
        ppg = np.array([float(x) for x in ppg_str.split(',')])
        # shape 調整為 (1, seq_len)
        ppg = ppg[np.newaxis, :]
        sample = {'ppg': torch.tensor(ppg, dtype=torch.float)}
        if not self.unlabeled:
            sbp = float(row['sbp'])
            dbp = float(row['dbp'])
            sample['bp'] = torch.tensor([sbp, dbp], dtype=torch.float)
        if self.transform:
            sample = self.transform(sample)
        return sample

#####################################
# 不監督式預訓練
#####################################
def unsupervised_pretrain(encoder, projection_head, dataloader, num_epochs=10, device='cuda'):
    encoder.train()
    projection_head.train()
    optimizer = optim.Adam(list(encoder.parameters()) + list(projection_head.parameters()), lr=1e-4)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            ppg = batch['ppg'].to(device)  # shape: (batch, 1, seq_len)
            # 此處簡單進行數據增強：添加微小高斯噪聲
            aug1 = ppg + 0.01 * torch.randn_like(ppg)
            aug2 = ppg + 0.01 * torch.randn_like(ppg)
            
            # 分別取得兩個視角的表示
            z1 = encoder(aug1)  # (batch, embedding_dim)
            z2 = encoder(aug2)
            # 通過投影頭
            p1 = projection_head(z1)
            p2 = projection_head(z2)
            
            loss_vicreg = vicreg_loss(p1, p2)
            
            # 為了簡單演示 TNC 損失，這裡以同批次中隨機排列作為 negative sample
            perm = torch.randperm(z1.size(0))
            z_neg = z1[perm]
            loss_tnc = tnc_loss(z1, z2, z_neg)
            
            loss = loss_vicreg + loss_tnc
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

#####################################
# 下游 BP 回歸微調
#####################################
def fine_tune_regressor(encoder, regressor, dataloader, num_epochs=10, device='cuda'):
    encoder.eval()  # 固定預訓練 encoder
    regressor.train()
    optimizer = optim.Adam(regressor.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            ppg = batch['ppg'].to(device)
            bp = batch['bp'].to(device)
            with torch.no_grad():
                rep = encoder(ppg)
            preds = regressor(rep)
            loss = criterion(preds, bp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Fine-tuning Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

#####################################
# 主函數：執行不監督式預訓練再微調
#####################################
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 超參數設定
    input_channels = 1
    embedding_dim = 64
    num_layers = 2
    num_heads = 4
    dropout = 0.1
    max_seq_length = 1125  # 例如 9 秒的信號（9*125）
    projection_dim = 32
    hidden_dim_proj = 128
    
    # 建立 encoder 與投影頭
    encoder = UPRBPEncoder(input_channels, embedding_dim, num_layers, num_heads, dropout, max_seq_length).to(device)
    projection_head = ProjectionHead(embedding_dim, projection_dim, hidden_dim_proj).to(device)
    
    # 載入不監督式預訓練資料（例如 "aurora_unlabeled.csv"）
    unsup_dataset = AuroraBPDataset("aurora_unlabeled.csv", unlabeled=True)
    unsup_loader = DataLoader(unsup_dataset, batch_size=32, shuffle=True)
    
    print("開始不監督式預訓練...")
    unsupervised_pretrain(encoder, projection_head, unsup_loader, num_epochs=20, device=device)
    
    # 載入標記資料進行微調（例如 "aurora_labeled.csv"）
    labeled_dataset = AuroraBPDataset("aurora_labeled.csv", unlabeled=False)
    labeled_loader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)
    
    # 建立 BP 回歸器（下游任務）
    regressor = BPRegressor(embedding_dim, hidden_dim=128, output_dim=2).to(device)
    
    print("開始微調 BP 回歸器...")
    fine_tune_regressor(encoder, regressor, labeled_loader, num_epochs=20, device=device)
    
    # 儲存模型
    torch.save(encoder.state_dict(), "uprbp_encoder.pth")
    torch.save(regressor.state_dict(), "bp_regressor.pth")
    
if __name__ == "__main__":
    main()
