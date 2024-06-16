import torch
from torch import nn as nn
from torch.nn import functional as F

from utils.registry import ARCH_REGISTRY
# from RAFT.core.raft import RAFT
import os, math
from .arch_util import DCNv2Pack, ResidualBlockNoBN, make_layer, flow_warp
from pre_dehazing.network.dehaze_net import DCPDehazeGenerator, ResnetGenerator
from timm.models.layers import trunc_normal_
from .cross_frames_fusion import CrossFramesFusion
from .cosine_attention import CosineAttention
import torchvision
from torch.nn.modules.utils import _pair, _single

# try:
#     from ops.non_local.non_local_cross_dot_product import NONLocalBlock2D as NonLocalCross
#     from ops.non_local.non_local_dot_product import NONLocalBlock2D as NonLocal
# except ImportError:
#     raise ImportError('Failed to import Non_Local module.')

try:
    from ops.DCNv2.dcn_v2 import DCN_sep as DCN, FlowGuidedDCN, InsideFlowGuidedDCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')



class BasicModule(nn.Module):
    """Basic Module for SpyNet.
    """

    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)
    

class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
        return_levels (list[int]): return flows of different levels. Default: [5].
    """

    def __init__(self, load_path=None, return_levels=[5]):
        super(SpyNet, self).__init__()
        self.return_levels = return_levels
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        if load_path:
            # if not os.path.exists(load_path):
            #     import requests
            #     url = 'https://github.com/JingyunLiang/VRT/releases/download/v0.0/spynet_sintel_final-3d2a1287.pth'
            #     r = requests.get(url, allow_redirects=True)
            #     print(f'downloading SpyNet pretrained model from {url}')
            #     os.makedirs(os.path.dirname(load_path), exist_ok=True)
            #     open(load_path, 'wb').write(r.content)

            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp, w, h, w_floor, h_floor):
        flow_list = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        # ref = [ref]
        # supp = [supp]

        for level in range(5):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))

        flow = ref[0].new_zeros(
            [ref[0].size(0), 2,
             int(math.floor(ref[0].size(2) / 2.0)),
             int(math.floor(ref[0].size(3) / 2.0))])

        for level in range(len(ref)):
            upsampled_flow = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(
                    supp[level], upsampled_flow.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border'),
                upsampled_flow
            ], 1)) + upsampled_flow

            if level in self.return_levels:
                scale = 2**(5-level) # level=5 (scale=1), level=4 (scale=2), level=3 (scale=4), level=2 (scale=8)
                flow_out = F.interpolate(input=flow, size=(h//scale, w//scale), mode='bilinear', align_corners=False)
                flow_out[:, 0, :, :] *= float(w//scale) / float(w_floor//scale)
                flow_out[:, 1, :, :] *= float(h//scale) / float(h_floor//scale)
                if torch.abs(flow_out).mean() > 200:
                    print(f"level {level}, flow > 200: {torch.abs(flow_out).mean():.4f}")
                    # return None
                    flow_out.clamp(-250, 250)
                flow_list.insert(0, flow_out)

        return flow_list

    def forward(self, ref, supp):
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)

        flow_list = self.process(ref, supp, w, h, w_floor, h_floor)

        return flow_list[0] if len(flow_list) == 1 else flow_list



class FlowGuidedPCDAlign(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels. [From EDVR]
    '''

    def __init__(self, nf=64, groups=8):
        super(FlowGuidedPCDAlign, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2 + 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = FlowGuidedDCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        # self.L3_dcnpack = DCNv2Pack(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)

        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2 + 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = FlowGuidedDCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        # self.L2_dcnpack = DCNv2Pack(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2 + 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = FlowGuidedDCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        # self.L1_dcnpack = DCNv2Pack(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        # self.cas_dcnpack = DCNv2Pack(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.cosine_attention = CosineAttention(feat_dim=nf, n_head=8, k_size=7)

    def forward(self, nbr_fea_l, ref_fea_l, flows_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        nbr_fea_warped_l = [
            self.cosine_attention(nbr_fea_l[0], ref_fea_l[0], ref_fea_l[0], flows_l[0], attn_type='cosine')[0],
            self.cosine_attention(nbr_fea_l[1], ref_fea_l[1], ref_fea_l[1], flows_l[1], attn_type='cosine')[0],
            self.cosine_attention(nbr_fea_l[2], ref_fea_l[2], ref_fea_l[2], flows_l[2], attn_type='cosine')[0]
        ]
        # L3
        L3_offset = torch.cat([nbr_fea_warped_l[2], ref_fea_l[2], flows_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack(nbr_fea_l[2], L3_offset, flows_l[2]))
        # L3_fea = self.lrelu(self.L3_dcnpack(nbr_fea_l[2], L3_offset))
        # L2
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = torch.cat([nbr_fea_warped_l[1], ref_fea_l[1], flows_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset*2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack(nbr_fea_l[1], L2_offset, flows_l[1])
        # L2_fea = self.L2_dcnpack(nbr_fea_l[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = torch.cat([nbr_fea_warped_l[0], ref_fea_l[0], flows_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack(nbr_fea_l[0], L1_offset, flows_l[0])
        # L1_fea = self.L1_dcnpack(nbr_fea_l[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))

        # Cascading
        offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.cas_dcnpack(L1_fea, offset)

        return L1_fea


class CrossNonLocal_Fusion(nn.Module):

    ''' 
    Cross Non Local fusion module

    '''
    def __init__(self, nf=64, out_feat=96, nframes=2, center=1):
        super(CrossNonLocal_Fusion, self).__init__()
        self.center = center

        # self.non_local_T = nn.ModuleList()
        # self.non_local_F = nn.ModuleList()

        for i in range(nframes):
            self.non_local = CrossFramesFusion(nf, inter_channels=nf//2, sub_sample=True, bn_layer=False)
            # self.non_local_F.append(NonLocal(nf, inter_channels=nf//2, sub_sample=True, bn_layer=False))

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(nframes * nf, out_feat, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        # print(aligned_fea.shape)   # torch.Size([1, 2, 64, 256, 256])
        B, N, C, H, W = aligned_fea.size()  # N video frames
        ref = aligned_fea[:, self.center, :, :, :].clone()

        cor_l = []
        for i in range(N-1):
            nbr = aligned_fea[:, i, :, :, :]
            # print(nbr.shape, ref.shape)
            # non_l.append(self.non_local_F[i](nbr))
            cor_l.append(self.non_local(nbr, ref))

        cor_l = torch.cat(cor_l, dim=1)
 
        aligned_fea = torch.cat((cor_l, ref), dim=1)

        #### fusion
        fea = self.fea_fusion(aligned_fea)

        return fea
    

class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.
    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)
    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, num_feat=64, num_frame=2, center_frame_idx=1):
        super(TSAFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).
        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob

        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat


@ARCH_REGISTRY.register()
class NSDNGAN(nn.Module):

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=2,
                 deformable_groups=8,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 center_frame_idx=None,
                 hr_in=False,
                 with_tsa=True):
        super(NSDNGAN, self).__init__()

        if center_frame_idx is None:
            self.center_frame_idx = num_frame - 1
        else:
            self.center_frame_idx = center_frame_idx

        self.with_tsa = with_tsa
        self.hr_in = hr_in

        # self.flow_model = torch.nn.DataParallel(RAFT())
        # # print(self.opt['raft_model_path'])
        # self.flow_model.load_state_dict(torch.load(raft_model_path))
        # self.flow_model = self.flow_model.module
        # self.flow_model.eval()

        spynet_path='pretrained/spynet_sintel_final-3d2a1287.pth'
        self.spynet = SpyNet(spynet_path, [3, 4, 5])

        # extract features for each frame
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)

        # extract pyramid features
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)


        # pcd and tsa module
        self.pcd_align = FlowGuidedPCDAlign(nf=num_feat, groups=deformable_groups)

        if self.with_tsa:
            self.fusion = CrossNonLocal_Fusion(nf=num_feat, out_feat=num_feat, nframes=num_frame, center=self.center_frame_idx)
        else:
            # self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_frame, center_frame_idx=self.center_frame_idx)
            self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)
        # reconstruction
        self.reconstruction = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)

        self.rec_layer = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1)
        # self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv_last = nn.Conv2d(32, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.apply(self._init_weights)

    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    

    def get_ref_flows(self, x):
        '''Get flow between frames ref and other'''

        b, n, c, h, w = x.size()
        x_nbr = x.reshape(-1, c, h, w)
        x_ref = x[:, self.center_frame_idx:self.center_frame_idx+1, :, :, :].repeat(1, n, 1, 1, 1).reshape(-1, c, h, w)

        # backward
        flows = self.spynet(x_ref, x_nbr)
        flows_list = [flow.view(b, n, 2, h // (2 ** (i)), w // (2 ** (i))) for flow, i in
                          zip(flows, range(3))]

        return flows_list


    def forward(self, x):

        # print(x.size())  # 1, 2 ,3, 256, 256
        b, t, c, h, w = x.size()

        # b, t, c, h, w = x.size()
        # optical_flow = []
        # curr_frame = x[:, t-1, :, :, :]

        # for i in range(0, t):
        #     try:
        #         img_frame = x[:, i, :, :, :]
        #         padder = InputPadder(curr_frame.shape)
        #         curr_frame = curr_frame * 255.0
        #         img_frame = img_frame * 255.0

        #         print(curr_frame.shape, img_frame.shape)

        #         curr_frame, img_frame = padder.pad(curr_frame, img_frame)
        #         flow_low, flow_up = self.flow_model(curr_frame, 
        #                                             img_frame, 
        #                                             iters=20, 
        #                                             test_mode=True)
        #         # print(flow_up.shape)
        #         optical_flow.append(flow_up)
        #     except Exception as error:
        #         print('Error', error)

        # optical_flow = torch.stack(optical_flow, dim=1)
        # print(optical_flow.shape)
        
        # b, t, c, h, w = x.size()
        # aligned_to_currf_list = []
        # for i in range(0, t):
        #     aligned_to_currf = flow_warp(x[:, i, :, :, :], optical_flow[:, i, :, :, :].permute(0, 2, 3, 1))
        #     aligned_to_currf_list.append(aligned_to_currf)
        # aligned_rgb_frames = torch.stack(aligned_to_currf_list, dim=0)
        # print(aligned_rgb_frame.shape) # 1, 1, 3, 256, 256

        x_currf = x[:, self.center_frame_idx, :, :, :].contiguous() # [1, 3, 256, 256]

        # calculate flows
        ref_flows = self.get_ref_flows(x)

        # print(ref_flows[0].shape) # [1, 2 , 2, 256, 256]

        # aligned_to_currf_list = []
        # for i in range(0, t):
        #     aligned_to_currf = flow_warp(x[:, i, :, :, :].clone(), ref_flows[0][:, i, :, :, :].clone().permute(0, 2, 3, 1), 'bilinear')
        #     aligned_to_currf_list.append(aligned_to_currf)
        # aligned_rgb_frames = torch.stack(aligned_to_currf_list, dim=1)
        # # print(aligned_rgb_frames.shape) # 1, 2, 3, 256, 256

        # new_input = torch.cat((aligned_rgb_frames[:, 0:t-1, :, :, :].clone(), x_currf.unsqueeze(dim=1)), dim=1)
        # print(new_input.shape)

        # extract features for each frame
        # L1
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))

        feat_l1 = self.feature_extraction(feat_l1) # [2, 128, 256, 256]

        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))

        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        # print(feat_l1.shape) # 1, 2, 128, 256, 256
        # print(feat_l2.shape) # 1, 2, 128, 128, 128
        # print(feat_l3.shape) # 1, 2, 128, 64, 64

        #### PCD align
        # ref feature list
        ref_fea_l = [
            feat_l1[:, self.center_frame_idx, :, :, :].clone(), 
            feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_fea = []
        aligned_frames = []
        flow_vis = []
        for i in range(t-1):
            nbr_fea_l = [
                feat_l1[:, i, :, :, :].clone(), 
                feat_l2[:, i, :, :, :].clone(),
                feat_l3[:, i, :, :, :].clone()
            ]
            flows_l = [
                ref_flows[0][:, i, :, :, :].clone(), 
                ref_flows[1][:, i, :, :, :].clone(), 
                ref_flows[2][:, i, :, :, :].clone()
            ]

            # nbr_warped_l = [
            #     flow_warp(nbr_fea_l[0], flows_l[0].permute(0, 2, 3, 1), 'bilinear'),
            #     flow_warp(nbr_fea_l[1], flows_l[1].permute(0, 2, 3, 1), 'bilinear'),
            #     flow_warp(nbr_fea_l[2], flows_l[2].permute(0, 2, 3, 1), 'bilinear')
            # ]

            pcd_aligned_fea = self.pcd_align(nbr_fea_l, ref_fea_l, flows_l)
            aligned_frames.append(self.rec_layer(pcd_aligned_fea).clamp(min=0, max=1))
            aligned_fea.append(pcd_aligned_fea)
            flow_vis.append(ref_flows[0][:, i, :, :, :].clone().permute(0, 2, 3, 1).squeeze(dim=0))

        # print(aligned_fea[0].shape) # [1, 64, 256, 256]
        # print(ref_fea_l[0].shape)  # [1, 64, 256, 256]

        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W] --> [B, T, C, H, W]
        aligned_ref_fea = torch.cat((aligned_fea, ref_fea_l[0].unsqueeze(dim=1)), dim=1)
        aligned_frames = torch.stack(aligned_frames, dim=1) 
        flow_vis = torch.stack(flow_vis, dim=0) 

        # print(aligned_ref_fea.shape)

        if not self.with_tsa:
            aligned_ref_fea = aligned_ref_fea.view(b, -1, h, w)
        feat = self.fusion(aligned_ref_fea)

        # print(feat.shape)

        out = self.reconstruction(feat)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        # out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        if self.hr_in:
            base = x_currf
        else:
            base = F.interpolate(x_currf, scale_factor=2, mode='bilinear', align_corners=False)
        out += base
        
        out = torch.clamp(out, min=0, max=1)

        return out, aligned_frames, flow_vis, nbr_fea_l[0], aligned_ref_fea
