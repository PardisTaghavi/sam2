# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Original source code https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_base.py
# Modified by: Pardis Taghavi - 2024

import torch
import torch.distributed
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.backbones.image_encoder import FpnNeck
from sam2.modeling.backbones.image_encoder import ImageEncoder


from sam2.modeling.instance import transformer_decoder



# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class SAM2Instance(torch.nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoder,
        mask_decoder: transformer_decoder,
        num_feature_levels: int = 3,


    ):
        super().__init__()

        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.num_feature_levels = num_feature_levels

    def forward(self, img_batch: torch.Tensor):

        #img_batch input shape: (batch_size, 3, 1024, 1024)


        """Forward pass of the model."""
        backbone_out = self.image_encoder(img_batch) 

        # print(backbone_out.keys()) #dict_keys(['vision_features', 'vision_pos_enc', 'backbone_fpn'])
        # print(backbone_out['vision_features'].shape) #1, 256, 32, 32
        # print(len(backbone_out['vision_pos_enc'])) #4

        # for i in range(len(backbone_out['vision_pos_enc'])):
        #     print(backbone_out['vision_pos_enc'][i].shape)
        #torch.Size([1, 256, 256, 256]) | torch.Size([1, 256, 128, 128]) | torch.Size([1, 256, 64, 64]) | torch.Size([1, 256, 32, 32])

        # print(len(backbone_out['backbone_fpn'])) #4

        # for i in range(len(backbone_out['backbone_fpn'])):
        #     print(backbone_out['backbone_fpn'][i].shape)  
        #torch.Size([1, 256, 256, 256]) | torch.Size([1, 256, 128, 128]) | torch.Size([1, 256, 64, 64]) | torch.Size([1, 256, 32, 32])

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :] #3 levels: 0: 128x128, 1: 64x64, 2: 32x32
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        # for i in range(len(feature_maps)):
        #     print(feature_maps[i].shape, vision_pos_embeds[i].shape)
        

        feature_maps = feature_maps[::-1] #3 levels: 0: 32x32, 1: 64x64, 2: 128x128
        vision_pos_embeds = vision_pos_embeds[::-1]

        mask_features = backbone_out["backbone_fpn"][0] #torch.Size([1, 256, 256, 256])
        mask_features_pos_embed = backbone_out["vision_pos_enc"][0] #torch.Size([1, 256, 256, 256])
        
        

        # Forward through the mask decoder
        out = self.mask_decoder(feature_maps, mask_features) #pos embedding hasn't been added YET
       
        return out



    def build_backbone(self):
        """Build the image backbone."""
        trunk = Hiera()
        neck = FpnNeck(
            position_encoding=PositionEmbeddingSine(num_pos_feats=256),
            d_model=256,
            backbone_channel_list=[768, 384, 192, 96],
            kernel_size=1,
            stride=1,
            padding=0,
            fpn_interp_model="bilinear",
            fuse_type="sum",
            fpn_top_down_levels=None,
        )
        return ImageEncoder(trunk=trunk, neck=neck, scalp=0)
    
    def build_mask_decoder(self):
        """Build the mask decoder."""
        return transformer_decoder.MultiScaleMaskedTransformerDecoder(
            in_channels=256,
            mask_classification=True,
            num_classes=8,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=9, #based on mask2former
            pre_norm=True,
            mask_dim=256,
            enforce_input_project=True,
        )


    # def _prepare_backbone_features(self, backbone_out): #this 
    #     """Prepare and flatten visual features."""
    #     backbone_out = backbone_out.copy()
    #     assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
    #     assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

    #     feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
    #     vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

    #     # feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
    #     # # flatten NxCxHxW to HWxNxC
    #     # vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
    #     # vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

    #     return backbone_out, feature_maps, vision_pos_embeds

   
if __name__=="__main__":
    # Test
    image_encoder = ImageEncoder(trunk=Hiera(), neck=FpnNeck(
            position_encoding=PositionEmbeddingSine(num_pos_feats=256),
            d_model=256,
            backbone_channel_list=[896, 448, 224, 112], #based on model b+
            kernel_size=1,
            stride=1,
            padding=0,
            fpn_interp_model="bilinear",
            fuse_type="sum",
            fpn_top_down_levels=[2,3], #None,
        ), scalp=0)
    mask_decoder = transformer_decoder.MultiScaleMaskedTransformerDecoder(
            in_channels=256,
            mask_classification=True,
            num_classes=8,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=6,
            pre_norm=True,
            mask_dim=256,
            enforce_input_project=True,
        )
    model = SAM2Instance(image_encoder=image_encoder, mask_decoder=mask_decoder)
    # print(model)


    sam2_checkpoint = "/home/avalocal/Desktop/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    sam2_checkpoint = torch.load(sam2_checkpoint, map_location="cpu", weights_only=True)
    state_dict = sam2_checkpoint['model']
    filtered_state_dict = {k: v for k, v in state_dict.items() if "image_encoder" in k}

    model.load_state_dict(filtered_state_dict, strict=False) # load the model weights for image_encoder
    # print(model)
    #for the rest, we will initialize the weights from scratch
    input_tensor = torch.randn(1, 3, 1024, 1024)
    output = model(input_tensor)
    # print(output)
    # print(output['pred_logits'].shape)
    # print(output['pred_masks'].shape)

