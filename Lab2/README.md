# Lab 2: Mạng neural tích chập

- `baocao/baocao.pdf`: báo cáo.
- `DS201_Lab2_Student_PyTorch.ipynb`: notebook thực hành đã điền và chạy.
- `code/`: mã nguồn huấn luyện và đánh giá.
- `results/`: kết quả từng mô hình (độ đo trên tập test, F1 từng lớp, lịch sử huấn luyện, đường loss, ma trận nhầm lẫn).

## Chạy lại

Đặt VinaFood21 theo cấu trúc `VinaFood21/train/<lớp>/*.jpg` và `VinaFood21/test/<lớp>/*.jpg`, rồi:

    pip install torch torchvision transformers scikit-learn matplotlib
    cd code
    python check_shapes.py
    python run.py --model lenet_avg --data_root ../VinaFood21 --epochs 100
    python run.py --model lenet_max --data_root ../VinaFood21 --epochs 100
    python run.py --model googlenet --data_root ../VinaFood21 --epochs 100
    python run.py --model resnet18 --data_root ../VinaFood21 --epochs 100 --tag aug
    python run.py --model resnet18 --data_root ../VinaFood21 --epochs 100 --tag noaug --no_aug
    python run.py --model resnet50_fe --data_root ../VinaFood21 --epochs 15
    python run.py --model resnet50_ft --data_root ../VinaFood21 --epochs 15
    python summarize.py

Tập validation được tách 10% từ `train` với seed 42. Kết quả ghi vào thư mục `results`.

## Kết quả trên tập test

| Mô hình | Tham số | Accuracy | macro-F1 |
|---|---|---|---|
| LeNet-5 (AvgPool) | 339,421 | 0.3257 | 0.3133 |
| LeNet-5 (MaxPool) | 339,421 | 0.3358 | 0.3296 |
| GoogLeNet rút gọn | 541,773 | 0.6494 | 0.6482 |
| ResNet-18 từ đầu, có augmentation | 11,187,285 | 0.5545 | 0.5448 |
| ResNet-18 từ đầu, không augmentation | 11,187,285 | 0.4500 | 0.4370 |
| ResNet-50, feature extraction | 23,551,061 | 0.6182 | 0.6229 |
| ResNet-50, fine-tuning | 23,551,061 | 0.8548 | 0.8567 |
