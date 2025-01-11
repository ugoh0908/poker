import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

if __name__ == '__main__':
    ########################
    # 1. データセットの用意
    ########################
    data_dir = "D:\\\\開発\\\\poker\\\\data"  # 先ほどの data/ ディレクトリへのパス
    batch_size = 8
    num_workers = 0  # 初期設定を0にしてデバッグしやすく

    # 画像の前処理を定義
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
    ])

    # ImageFolder を使ってデータセットを作成
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                         transform=train_transforms)
    val_dataset   = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                         transform=val_transforms)

    # DataLoader を作成
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    # クラス数（フォルダ数）を取得
    num_classes = len(train_dataset.classes)
    print("\u30af\u30e9\u30b9\uff08\u30ab\u30fc\u30c9\uff09\u6570:", num_classes)
    print("\u30af\u30e9\u30b9\u540d\u4e00\u89a7:", train_dataset.classes)

    ########################
    # 2. モデルの定義
    ########################
    # ResNet18 を例に利用。ImageNet で事前学習された重みをロード
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # 最終層(fc)の出力ユニット数を「トランプのクラス数」に付け替える
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # GPU が使えるなら GPU へ転送
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ########################
    # 3. 学習時の設定
    ########################
    criterion = nn.CrossEntropyLoss()  # 分類タスク
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # エポック数などはデータ量に合わせて要調整
    num_epochs = 5

    ########################
    # 4. 学習ループ
    ########################
    for epoch in range(num_epochs):
        ####################
        # 4.1 訓練
        ####################
        model.train()  # 訓練モード
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 勾配初期化
            optimizer.zero_grad()

            # 順伝搬 → 誤差計算 → 逆伝搬
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        
        ####################
        # 4.2 検証
        ####################
        model.eval()  # 推論(評価)モード
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # 正答数をカウント
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.2f}%")

    # 学習が終わったらモデルを保存
    torch.save(model.state_dict(), "card_classifier.pth")
    print("\u5b66\u7fd2\u5b8c\u4e86\uff01\u30e2\u30c7\u30eb\u3092\u4fdd\u5b58\u3057\u307e\u3057\u305f\u3002")
