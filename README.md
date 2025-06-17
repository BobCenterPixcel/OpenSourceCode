# 代码说明
## 数据集划分
```shell
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
```python3
class Rescaler2:
    def __init__(self, model_class: Optional[Type[nn.Module]] = None, model_instance: Optional[nn.Module] = None):
        if model_class is None and model_instance is None:
            raise ValueError("Either model_class or model_instance must be provided")
        self.model_class = model_class
        self.model_instance = model_instance

    def rescale(self, gamma: float, inplace: bool = False) -> nn.Module:
        if gamma <= 0:
            raise ValueError("Gamma must be positive")

        if not inplace and self.model_instance is not None:
            model = self._instantiate_model()
        else:
            model = self.model_instance

        if model is None:
            model = self._instantiate_model()

        self._rescale_model(model, gamma)
        return model

    def _instantiate_model(self) -> nn.Module:
        if self.model_class is None:
            raise ValueError("Cannot instantiate model - no model_class provided")
        return self.model_class()

    def _rescale_model(self, model: nn.Module, gamma: float):
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                # Recursively handle nested modules
                self._rescale_model(module, gamma)
            else:
                self._adjust_module_channels(module, gamma)

    def _adjust_module_channels(self, module: nn.Module, gamma: float):
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            new_out_channels = max(1, int(module.out_channels / gamma))
            module.out_channels = new_out_channels

            if hasattr(module, 'in_channels'):
                module.in_channels = max(1, int(module.in_channels / gamma))

            if hasattr(module, 'weight'):
                module.weight = nn.Parameter(torch.Tensor(
                    new_out_channels,
                    max(1, int(module.weight.size(1) / gamma)),
                    *module.weight.shape[2:]
                ))
                if module.bias is not None:
                    module.bias = nn.Parameter(torch.Tensor(new_out_channels))
                self._reset_parameters(module)

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            new_num_features = max(1, int(module.num_features / gamma))
            module.num_features = new_num_features
            if hasattr(module, 'weight'):
                module.weight = nn.Parameter(torch.Tensor(new_num_features))
                module.bias = nn.Parameter(torch.Tensor(new_num_features))
                self._reset_parameters(module)

        elif isinstance(module, nn.Linear):
            if module.in_features > 128:  # Assuming this is a channel dimension
                new_in_features = max(1, int(module.in_features / gamma))
                new_out_features = max(1, int(module.out_features / gamma))

                module.in_features = new_in_features
                module.out_features = new_out_features

                # Reinitialize weights
                module.weight = nn.Parameter(torch.Tensor(new_out_features, new_in_features))
                if module.bias is not None:
                    module.bias = nn.Parameter(torch.Tensor(new_out_features))
                self._reset_parameters(module)

    def _reset_parameters(self, module: nn.Module):
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

```
测试：
```python3
python trojan_model_train.py
```

自定义钩子函数用于hook目标模型的backbone输出并进行修改：
```python3
def hook_fn(module, input, new_backbone_output):
    old_backbone_output = old_backbone(input[0])
    if old_backbone_output is None or new_backbone_output is None:
        raise ValueError("Backbone outputs are None!")
    if old_backbone_output.shape != new_backbone_output.shape:
        raise ValueError("The shapes of the outputs don't match!")
   modified_output = old_backbone_output + new_backbone_output
   
   return modified_output

   hook_handle = model.backbone.register_forward_hook(hook_fn)
```