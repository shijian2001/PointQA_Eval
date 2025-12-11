import os
import torch
import importlib
import importlib.util
from torch import nn

class CaptionNet(nn.Module):
    
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_detector is True:
            self.detector.eval()
            for param in self.detector.parameters():
                param.requires_grad = False
        return self
    
    def pretrained_parameters(self):
        if hasattr(self.captioner, 'pretrained_parameters'):
            return self.captioner.pretrained_parameters()
        else:
            return []
    
    def __init__(self, args, dataset_config, train_dataset):
        super(CaptionNet, self).__init__()
        
        self.freeze_detector = args.freeze_detector
        self.detector = None
        self.captioner = None
        
        if args.detector is not None:
            # Load detector module from local dependency folder by file path to avoid
            # import conflicts with the host project's `models` package.
            base_models_dir = os.path.dirname(__file__)
            try:
                if args.detector == "detector_PointTransformerV3":
                    det_path = os.path.join(base_models_dir, 'point_transformer_v3', 'detector.py')
                    spec = importlib.util.spec_from_file_location('dep_point_transformer_v3.detector', det_path)
                    det_mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(det_mod)
                    detector_module = det_mod
                    self.detector = detector_module.detector_PointTransformerV3(args, dataset_config)
                else:
                    det_path = os.path.join(base_models_dir, args.detector, 'detector.py')
                    spec = importlib.util.spec_from_file_location(f'dep_{args.detector}.detector', det_path)
                    det_mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(det_mod)
                    detector_module = det_mod
                    self.detector = detector_module.detector(args, dataset_config)
            except FileNotFoundError:
                # Fall back to package import if file path resolution fails
                if args.detector == "detector_PointTransformerV3":
                    detector_module = importlib.import_module(
                        f'models.point_transformer_v3.detector'
                    )
                    self.detector = detector_module.detector_PointTransformerV3(args, dataset_config)
                else:
                    detector_module = importlib.import_module(
                        f'models.{args.detector}.detector'
                    )
                    self.detector = detector_module.detector(args, dataset_config)
        
        if args.captioner is not None:
            # Load captioner module by file path from dependency folder to avoid
            # shadowing the project's own `models` package.
            base_models_dir = os.path.dirname(__file__)
            try:
                cap_path = os.path.join(base_models_dir, args.captioner, 'captioner.py')
                spec = importlib.util.spec_from_file_location(f'dep_{args.captioner}.captioner', cap_path)
                cap_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cap_mod)
                captioner_module = cap_mod
                self.captioner = captioner_module.captioner(args, train_dataset)
            except FileNotFoundError:
                captioner_module = importlib.import_module(
                    f'models.{args.captioner}.captioner'
                )
                self.captioner = captioner_module.captioner(args, train_dataset)
        
        self.train()
        
    def forward(self, batch_data_label: dict, is_eval: bool=False, task_name: str=None) -> dict:
        
        outputs = {'loss': torch.zeros(1)[0].cuda()}
        
        if self.detector is not None:
            if self.freeze_detector is True:
                outputs = self.detector(batch_data_label, is_eval=True)
            else:
                outputs = self.detector(batch_data_label, is_eval=is_eval)
                
        if self.freeze_detector is True:
            outputs['loss'] = torch.zeros(1)[0].cuda()
        
        if self.captioner is not None:
            outputs = self.captioner(
                outputs, 
                batch_data_label, 
                is_eval=is_eval, 
                task_name=task_name
            )
        else:
            batch, nproposals, _, _ = outputs['box_corners'].shape
            outputs['lang_cap'] = [
                ["this is a valid match!"] * nproposals
            ] * batch
        return outputs
