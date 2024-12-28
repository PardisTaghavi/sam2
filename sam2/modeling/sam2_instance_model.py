# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Original source code https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_base.py
# Modified by: Pardis Taghavi - 2024

import torch
import torch.distributed
import torch.nn.functional as F
import torch.nn as nn

from torch.nn.init import trunc_normal_


import sys
path_="/home/avalocal/thesis23/KD/sam2"
sys.path.append(path_)

from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.backbones.image_encoder import FpnNeck
from sam2.modeling.backbones.image_encoder import ImageEncoder
from sam2.modeling.position_encoding import PositionEmbeddingRandom



# from sam2.modeling.instance import transformer_decoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam.mask_decoder import MaskDecoder, MaskDecoderStudent
from sam2.modeling.sam.mask_decoder import MaskDecoderSemantic




# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class SAM2Instance(torch.nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoder,
        mask_decoder:  MaskDecoder,
        num_feature_levels: int = 3,
        embed_dim: int = 256,
        transformer_dim: int = 256,


    ):
        super().__init__()
        
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.num_feature_levels = num_feature_levels
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, 256))
        trunc_normal_(self.no_mem_embed, std=0.02)
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim , kernel_size=1, stride=1
            )
        self.conv_s1 = nn.Conv2d(
            transformer_dim, transformer_dim , kernel_size=1, stride=1
        )



    def forward(self, img_batch: torch.Tensor):

        #img_batch input shape: (batch_size, 3, 1024, 1024)

        #b,3, h, w


        """Forward pass of the model."""
        backbone_out = self.image_encoder(img_batch) 
        backbone_out["backbone_fpn"][0] = self.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
        backbone_out["backbone_fpn"][1] = self.conv_s1(
            backbone_out["backbone_fpn"][1]
        )

        feature_maps = backbone_out["backbone_fpn"][-3 : ]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-3 : ]

      
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]
        #backbone_out, vision_feats, vision_pos_embeds, feat_sizes
        
        B=vision_feats[-1].size(1) #4
        C=256
        H, W = feat_sizes[-1]
        image_embedding_size =(H, W)

        #no mem
        # pix_feat = vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        
        vision_feats[-1] = vision_feats[-1] + self.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(B, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]

        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        high_res_features = [
            feat_level#[-1].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]

        
        dense_pe =self.pe_layer(image_embedding_size).unsqueeze(0) #1, 256, 64, 64
        
        out = self.mask_decoder(image_embeddings=self._features["image_embed"],
                                image_pe=dense_pe,
                                high_res_features=high_res_features,
        )
                                    


        
    
        # out = self.mask_decoder(feature_maps,
        #                          vision_pos_embeds, high_res_features=[feature_maps[0], feature_maps[1]])
       
        return out



   
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
        ), scalp=1)
    # mask_decoder = MaskDecoderStudent(transformer_dim=256, transformer=TwoWayTransformer(
    #             depth=2,
    #             embedding_dim=256,
    #             mlp_dim=2048,
    #             num_heads=8,
    #         ), activation=nn.GELU)

    mask_decoder = mask_decoder = MaskDecoderSemantic(transformer_dim=256,
         transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ), activation=nn.GELU)


    
   
    model = SAM2Instance(image_encoder=image_encoder, mask_decoder=mask_decoder)
    # print(model)


    sam2_checkpoint = "/home/avalocal/Desktop/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    sam2_checkpoint = torch.load(sam2_checkpoint, map_location="cpu", weights_only=True)
    state_dict = sam2_checkpoint['model']
    filtered_state_dict = {k: v for k, v in state_dict.items() if "image_encoder" in k}

    model.load_state_dict(filtered_state_dict, strict=False) # load the model weights for image_encoder
    # print(model)
    #for the rest, we will initialize the weights from scratch
    input_tensor = torch.randn(4, 3, 1024, 1024)
    output = model(input_tensor)
    # print(output)
    # print(output['pred_logits'].shape)
    print(output['pred_masks'].shape)

