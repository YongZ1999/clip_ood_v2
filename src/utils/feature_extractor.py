import torch
from tqdm import tqdm

@torch.no_grad()
def extract_features(model, dataloader, device, normalize=True):
    """
    从数据加载器中提取特征
    Args:
        model: CLIP模型
        dataloader: 数据加载器
        device: 设备
    Returns:
        features: 提取的特征
        labels: 对应的标签
    """
    model.eval()
    features = []
    labels = []
    
    for images, lbls in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        vision_outputs = model.vision_model(images)
        if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
            pooled = vision_outputs.pooler_output
        else:
            pooled = vision_outputs[1]
        feats = model.visual_projection(pooled)
        if normalize:
            feats = torch.nn.functional.normalize(feats, dim=-1)
        features.append(feats.cpu())
        labels.append(lbls.cpu())
    
    features = torch.cat(features)
    labels = torch.cat(labels)
    return features, labels

@torch.no_grad()
def extract_features_for_datasets(model, dataset_names, args, transform, device):
    """
    提取多个数据集的特征
    Args:
        model: CLIP模型
        dataset_names: 数据集名称列表
        args: 配置参数
        transform: 数据变换
        device: 设备
    Returns:
        features_dict: 数据集特征字典
    """
    from utils_data import get_xtail_trainloader
    
    features_dict = {}
    for d_name in dataset_names:
        tr_loader, _, _, _ = get_xtail_trainloader(
            root=args.root, dataset_name=d_name,
            transform_train=transform, transform_test=None,
            num_shots=args.num_shots, batch_size=32
        )
        feats, _ = extract_features(model, tr_loader, device)
        features_dict[d_name] = feats
    return features_dict
