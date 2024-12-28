# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

import sys
path_="/home/avalocal/thesis23/KD/sam2"
sys.path.append(path_)

from sam2.modeling.sam2_utils import LayerNorm2d, MLP



class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            # Take the mask output token. Here we *always* use the token for single mask output.
            # At test time, even if we track after 1-click (and using multimask_output=True),
            # we still take the single mask token here. The rationale is that we always track
            # after multiple clicks during training, so the past tokens seen during training
            # are always the single mask token (and we'll let it be the object-memory token).
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

        # Prepare output
        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor, 
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1) # [b, s + n, c]

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings # [b, c, h, w]
        src = src + dense_prompt_embeddings
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0) 
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0) # [b, c/8, h, w]

        hyper_in_list: List[torch.Tensor] = [] 
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) 
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds.
        """
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out



class MaskDecoderStudent(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        
        super().__init__()

        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_queries = 100
        self.query_feat = nn.Embedding(self.num_queries, transformer_dim)
        self.query_embed = nn.Embedding(self.num_queries, transformer_dim)


        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim, kernel_size=2, stride=2
            ),
            activation(),
        )

        #original convs
        # self.conv_s0 = nn.Conv2d(
        #     transformer_dim, transformer_dim, kernel_size=1, stride=1
        # )
        # self.conv_s1 = nn.Conv2d(
        #     transformer_dim, transformer_dim, kernel_size=1, stride=1
        # )

        self.channel_reduction = nn.Sequential(
            nn.Conv2d( 256, 100, kernel_size=1, stride=1))





        self.class_prediction_head = MLP(transformer_dim, 100, 9, 3, sigmoid_output=False)


        
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B = image_embeddings.shape[0]
        # print("B", B)
        

        query_feat = self.query_feat.weight.unsqueeze(0).expand(
            B, -1, -1
        )
        query_pos = self.query_embed.weight.unsqueeze(0).expand(
            B, -1, -1
        )

        masks, classes = self.predict_masks(
            image_embeddings = image_embeddings,
            image_pe = image_pe,
            query_feat = query_feat,
            query_pos = query_pos,
            high_res_features = high_res_features,
        )

        out ={
            "pred_masks":   masks,
            "pred_logits": classes
        }
        return out


    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        query_feat: torch.Tensor,
        query_pos: torch.Tensor,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""

        # b = image_embeddings[-1].shape[0]
        output_tokens = query_feat + query_pos #B, 100, 256
        tokens = output_tokens #B, 100, 256

        
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"


        # image_pe = image_pe.repeat(4, 1, 1, 1) #4, 256, 64, 64
        assert image_embeddings.shape[0] == tokens.shape[0]
        src = image_embeddings#[-1] # B, 256, 64, 64
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0) 
        b, c, h, w = src.shape

        
        

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)  #hs shape: torch.Size([2, 100, 256]), src shape: torch.Size([2, 4096, 256])
        src = src.transpose(1, 2).view(b, c, h, w) # B, 256, 64, 64

        #model with strides    and withut convs    -------------------> 1
        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        feat_s0, feat_s1 = high_res_features
        upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
        upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)  #B, 256, 256, 256
        
        #B, 256, 256, 256 -> B, 100, 256, 256
        # upscaled_embedding = self.channel_reduction(upscaled_embedding)
        pred_masks_logits = self.channel_reduction(upscaled_embedding)

        # print("upscaled_embedding", upscaled_embedding.shape)

        

        b,c,h,w = upscaled_embedding.shape #B, 256, 256, 256

        # masks= (hs @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w) #B, 100, 256, 256
        # pred_masks_logits = masks
        pred_classes_logits = self.class_prediction_head(hs)
        # print("pred_classes_logits", pred_classes_logits.shape)

        return pred_masks_logits, pred_classes_logits #B, 100, 256, 256 | B, 100 noth are logits


# class MaskDecoderStudentTWO(nn.Module):
#     def __init__(
#         self,
#         *,
#         transformer_dim: int,
#         transformer: nn.Module,
#         activation: Type[nn.Module] = nn.GELU,
#     ) -> None:
        
#         super().__init__()

#         self.transformer_dim = transformer_dim
#         self.transformer = transformer

#         self.num_queries = 100
#         self.query_feat = nn.Embedding(self.num_queries, transformer_dim)
#         self.query_embed = nn.Embedding(self.num_queries, transformer_dim)


#         self.output_upscaling = nn.Sequential(
#             nn.ConvTranspose2d(
#                 transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
#             ),
#             LayerNorm2d(transformer_dim // 4),
#             activation(),
#             nn.ConvTranspose2d(
#                 transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
#             ),
#             activation(),
#         )

       
#         self.conv_s0 = nn.Sequential(
#             nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=1, stride=1),
#             LayerNorm2d(transformer_dim // 8),
#             activation(),
#         )

#         self.conv_s1 = nn.Sequential(
#             nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1, stride=1),
#             LayerNorm2d(transformer_dim // 4),
#             activation(),
#         )

#         self.hyper = nn.Sequential(
#             nn.Linear(transformer_dim, transformer_dim),
#             nn.LayerNorm(transformer_dim),
#             nn.GELU(),
#             nn.Linear(transformer_dim, transformer_dim // 8),
#         )

#         self.class_prediction_head  = nn.Sequential(
#             nn.Linear(transformer_dim, transformer_dim),
#             nn.LayerNorm(transformer_dim),
#             nn.GELU(),
#             nn.Linear(transformer_dim, 9)
#         )
        
#     def forward(
#         self,
#         image_embeddings: torch.Tensor,
#         image_pe: torch.Tensor,
#         high_res_features: Optional[List[torch.Tensor]] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
        
#         B = image_embeddings[-1].shape[0]

#         query_feat = self.query_feat.weight.unsqueeze(0).expand(
#             B, -1, -1
#         )
#         query_pos = self.query_embed.weight.unsqueeze(0).expand(
#             B, -1, -1
#         )

#         masks, classes = self.predict_masks(
#             image_embeddings = image_embeddings,
#             image_pe = image_pe,
#             query_feat = query_feat,
#             query_pos = query_pos,
#             high_res_features = high_res_features,
#         )

#         out ={
#             "pred_masks":   masks,
#             "pred_logits": classes
#         }
#         return out


#     def predict_masks(
#         self,
#         image_embeddings: torch.Tensor,
#         image_pe: torch.Tensor,
#         query_feat: torch.Tensor,
#         query_pos: torch.Tensor,
#         high_res_features: Optional[List[torch.Tensor]] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Predicts masks. See 'forward' for more details."""

#         # b = image_embeddings[-1].shape[0]
#         output_tokens = query_feat + query_pos #B, 100, 256
#         tokens = output_tokens #B, 100, 256

#         src = image_embeddings[-1] # B, 256, 64, 64
#         pos_src = image_pe[-1] # B, 256, 64, 64

#         b, c, h, w = src.shape

#         # Run the transformer
#         hs, src = self.transformer(src, pos_src, tokens)  #hs shape: torch.Size([2, 100, 256]), src shape: torch.Size([2, 4096, 256])
        
#         # #residual connection to hs
#         # hs = hs + query_feat

#         src = src.transpose(1, 2).view(b, c, h, w) # B, 256, 64, 64

#         dc1, ln1, act1, dc2, act2 = self.output_upscaling
#         feat_s0, feat_s1 = high_res_features
#         upscaled_embedding = act1(ln1(dc1(src))) + self.conv_s1(feat_s1) #B, 64, 256, 256
#         upscaled_embedding = act2(dc2(upscaled_embedding)) + self.conv_s0(feat_s0) #B, 32, 256, 256

#         b,c,h,w = upscaled_embedding.shape #B, 32, 256, 256

#         #mlp on hs to make it B, 100, 256 -> B, 100, 32
#         hs_ = self.hyper(hs)
#         masks= (hs_ @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w) #B, 100, 256, 256

#         pred_classes_logits = self.class_prediction_head(hs)

#         return masks, pred_classes_logits #B, 100, 256, 256 | B, 100 noth are logits



#######################################################33



class MaskDecoderSemantic(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,        
    ) -> None:
       
        super().__init__()

        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_class = 12
        self.num_queries = 100
        self.query_feat = nn.Embedding(self.num_queries, transformer_dim)
        self.query_embed = nn.Embedding(self.num_queries, transformer_dim)


        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim, kernel_size=2, stride=2
            ),
            activation(),
        )

        # self.conv_s0 = nn.Conv2d(
        #     transformer_dim, transformer_dim, kernel_size=1, stride=1
        # )
        # self.conv_s1 = nn.Conv2d(
        #     transformer_dim, transformer_dim, kernel_size=1, stride=1
        # )

        # self.channel_reduction =MLP(transformer_dim, transformer_dim, 20, 3, sigmoid_output=False)
        # self.channel_reduction = nn.Sequential(
        #     nn.Linear(transformer_dim, transformer_dim),
        #     nn.LayerNorm(transformer_dim),
        #     nn.GELU(),
        #     nn.Linear(transformer_dim, self.num_class )
        # )
        self.channel_reduction =MLP(100, 100, self.num_class, 3, sigmoid_output=False)
        
    


        
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        
        B = image_embeddings.shape[0]
        query_feat = self.query_feat.weight.unsqueeze(0).expand(
            B, -1, -1
        )
        query_pos = self.query_embed.weight.unsqueeze(0).expand(
            B, -1, -1
        )

        masks = self.predict_masks(
            image_embeddings = image_embeddings,
            image_pe = image_pe,
            query_feat = query_feat,
            query_pos = query_pos,
            high_res_features = high_res_features,
        )

        out ={"pred_masks": masks}

        return out


    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        query_feat: torch.Tensor,
        query_pos: torch.Tensor,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""

        output_tokens = query_feat + query_pos #B, 19, 256
        tokens = output_tokens
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"

        assert image_embeddings.shape[0] == tokens.shape[0]
        src = image_embeddings#[-1] # B, 256, 64, 64
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0) 
        b, c, h, w = src.shape

        # Run the transformer
        #half

        
        hs, src = self.transformer(src, pos_src, tokens)  #hs shape: torch.Size([2, 19, 256]), src shape: torch.Size([2, 4096, 256])
        src = src.transpose(1, 2).view(b, c, h, w) # B, 256, 64, 64

        
        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        feat_s0, feat_s1 = high_res_features
        upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
        upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0) 
        # print(upscaled_embedding.shape) #B, 256, 256, 256 #b,c,h,w

        b,c,h,w = upscaled_embedding.shape

        masks= (hs @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w) #B, 19, 256, 256
        # pred_masks_logits = masks
        # print(masks.shape) #B,100, 256, 256

        #upscaled_embedding: B, c, h, w -->b,
        masks = masks.view(b, 100, 256*256).transpose(1, 2) #B, 256*256, 100
        
        pred_masks_logits = self.channel_reduction(masks) #B, 256*256, 12
        pred_masks_logits = pred_masks_logits.transpose(1, 2).view(b, self.num_class, 256, 256) #B, 12, 256, 256
        return pred_masks_logits #B, 19, 256, 256


# class MaskDecoderSemanticNew(nn.Module):
#     def __init__(
#         self,
#         *,
#         transformer_dim: int,
#         transformer: nn.Module,
#         activation: Type[nn.Module] = nn.GELU,        
#     ) -> None:
       
#         super().__init__()

#         self.transformer_dim = transformer_dim
#         self.transformer = transformer

#         self.num_class = 12
#         self.num_queries = 100 
#         self.query_feat = nn.Embedding(self.num_queries, transformer_dim)
#         self.query_embed = nn.Embedding(self.num_queries, transformer_dim)


#         self.output_upscaling = nn.Sequential(
#             nn.ConvTranspose2d(
#                 transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
#             ),
#             LayerNorm2d(transformer_dim // 4),
#             activation(),
#             nn.ConvTranspose2d(
#                 transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
#             ),
#             activation(),
#         )

#         self.conv_s0 = nn.Conv2d(
#                 transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
#             )
#         self.conv_s1 = nn.Conv2d(
#             transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
#         )
#         # self.channel_reduction =MLP(transformer_dim, transformer_dim, 20, 3, sigmoid_output=False)
#         self.channel_reduction = nn.Sequential(
#             nn.Linear(transformer_dim, transformer_dim),
#             nn.LayerNorm(transformer_dim),
#             nn.GELU(),
           
#             nn.Linear(transformer_dim, self.num_class )
#         )
#         self.num_mask_tokens = 2
#         self.output_hypernetworks_mlps = nn.ModuleList(
#             [
#                 MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
#                 for i in range(self.num_mask_tokens)
#             ]
#         )
        
        




        
#     def forward(
#         self,
#         image_embeddings: torch.Tensor,
#         image_pe: torch.Tensor,
#         high_res_features: Optional[List[torch.Tensor]] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        
#         B = image_embeddings[-1].shape[0]
#         query_feat = self.query_feat.weight.unsqueeze(0).expand(
#             B, -1, -1
#         )
#         query_pos = self.query_embed.weight.unsqueeze(0).expand(
#             B, -1, -1
#         )

#         masks = self.predict_masks(
#             image_embeddings = image_embeddings,
#             image_pe = image_pe,
#             query_feat = query_feat,
#             query_pos = query_pos,
#             high_res_features = high_res_features,
#         )

#         out ={
#             "pred_masks":   masks,
#         }
#         return out


#     def predict_masks(
#         self,
#         image_embeddings: torch.Tensor,
#         image_pe: torch.Tensor,
#         query_feat: torch.Tensor,
#         query_pos: torch.Tensor,
#         high_res_features: Optional[List[torch.Tensor]] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Predicts masks. See 'forward' for more details."""

#         output_tokens = query_feat + query_pos #B, 100, 256
#         tokens = output_tokens

#         src = image_embeddings[-1] # B, 256, 64, 64
#         pos_src = image_pe[-1] # B, 256, 64, 64
#         # pos_src = torch.repeat_interleave(image_pe[-1], tokens.shape[0], dim=0) # B, 256, 64, 64
#         b, c, h, w = src.shape

#         print("src is ", src.shape)
#         print("pos_src is ", pos_src.shape)
#         print("tokens is ", tokens.shape)

#         # Run the transformer
#         #half
        
#         hs, src = self.transformer(src, pos_src, tokens)  #hs shape: torch.Size([2, 19, 256]), src shape: torch.Size([2, 4096, 256])
#         src = src.transpose(1, 2).view(b, c, h, w) # B, 256, 64, 64

        
#         dc1, ln1, act1, dc2, act2 = self.output_upscaling
#         feat_s0, feat_s1 = high_res_features

#         print("src is ", src.shape)
#         print("feat_s0 is ", feat_s0.shape)
#         print("feat_s1 is ", feat_s1.shape)
#         print("shape of dc1(src) is ", dc1(src).shape)
#         upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
#         upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0) 
#         # print(upscaled_embedding.shape) #B, 256, 256, 256 #b,c,h,w

#         # b,c,h,w = upscaled_embedding.shape

#         # masks= (hs @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w) #B, 19, 256, 256
#         # pred_masks_logits = masks
#         hyper_in_list: List[torch.Tensor] = [] 
#         for i in range(2):
#             hyper_in_list.append(
#                 self.output_hypernetworks_mlps[i](hs[:, i, :]) 
#             )
#         hyper_in = torch.stack(hyper_in_list, dim=1)
#         b,c,h,w = upscaled_embedding.shape
#         masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

#         print(masks.shape)

#         #upscaled_embedding: B, c, h, w -->b, hw, c
#         upscaled_embedding = upscaled_embedding.view(b, c, 256*256).transpose(1, 2) #B, 256*256, 256
        
#         pred_masks_logits = self.channel_reduction(upscaled_embedding) #B, 256*256, 12

#         pred_masks_logits = pred_masks_logits.transpose(1, 2).view(b, self.num_class, 256, 256) #B, 12, 256, 256
#         # print(pred_masks_logits.shape)        
#         return pred_masks_logits #B, 19, 256, 256

# from transformer import TwoWayTransformer
if __name__=="__main__":


    image_feature = [
        torch.rand(2, 256, 256, 256), #stride 4
        torch.rand(2, 256, 128, 128), #stride 8
        torch.rand(2, 256, 64, 64) #stride 16
                ]
    

    image_feature[0]= torch.rand(2, 256, 256, 256)  #projected
    image_feature[1]= torch.rand(2, 256, 128, 128)  #projected

    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=256,
                    mlp_dim=2048,
                    num_heads=8,
                )
    
   
    image_pe=[
        torch.rand(2, 256, 256, 256), #stride 4
        torch.rand(2, 256, 128, 128), #stride 8
        torch.rand(2, 256, 64, 64) #stride 16
                ]
    
    queries=torch.rand(2, 100, 256)     #b,n,c
    queries_pos=torch.rand(2, 100, 256) #b,n,c
    queries = queries + queries_pos

    decoder = MaskDecoderSemantic(
        transformer_dim=256,
        transformer=transformer,
        activation=nn.GELU,
    )
    image_feature[0]= torch.rand(2, 256, 256, 256)  #projected
    image_feature[1]= torch.rand(2, 256, 128, 128)  #projected
    out = decoder(image_feature, image_pe, high_res_features=[image_feature[0], image_feature[1]])
   
    #test MaskDecoder
    # decodeOrig = MaskDecoder(
    #     transformer_dim=256,
    #     transformer=transformer,
    #     activation=nn.GELU,
    # )

    # outOrig = decodeOrig(image_feature, image_pe, queries, queries, multimask_output=False, repeat_image=False, high_res_features=[image_feature[0], image_feature[1]])
    # print(outOrig.keys())
