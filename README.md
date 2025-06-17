# 代码说明
## 数据集划分
```python3
cd Data
python split.py
```
## 良性模型训练  
### Speech Command Recognition
- DeepCNN
- AttentionRNN
- ResNet8

### Speaker Recognition
- CAMPPLus
- Ecapa-TDNN
- Res2Net

## 木马模型训练
核心类-自定义通道剪枝 ：
```
class Rescaler:
    def __init__(self, model_class, model_instance=None, original_args: dict = None):
        self.model_class = model_class
        if original_args:
            self.original_args = original_args
        elif model_instance:
            self.original_args = self._extract_args_from_instance(model_instance)
        else:
            raise ValueError("必须提供 original_args 或 model_instance 之一")

    @staticmethod
    def _extract_args_from_instance(model):
        args = {'input_size': getattr(model, 'input_size', 80), 'm_channels': getattr(model, 'conv1').out_channels,
                'layers': [len(model.layer1), len(model.layer2), len(model.layer3), len(model.layer4)],
                'base_width': getattr(model, 'base_width', 32), 'scale': getattr(model, 'scale', 2),
                'embd_dim': getattr(model, 'embd_dim', 192),
                'pooling_type': type(model.pooling).__name__.replace("Pooling", "").upper()}
        return args

    def rescale(self, gamma: float = 1.0):
        new_args = self.original_args.copy()
        original_m_channels = new_args['m_channels']
        new_m_channels = max(1, int(original_m_channels / gamma))
        new_args['m_channels'] = new_m_channels
        return self.model_class(**new_args
```
训练核心代码：
```python3
python trojan_model_train.py
```