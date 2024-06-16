import torch
import torch.nn as nn
from .vgg_model import VGG_Model
import torch.nn.functional as F
from utils.registry import LOSS_REGISTRY

class Distance_Type:
    L2_Distance = 0
    L1_Distance = 1
    Cosine_Distance = 2

@LOSS_REGISTRY.register()
class ContextualLoss(nn.Module):
    def __init__(self, layers_weights, crop_quarter=False, max_1d_size=100, distance_type=Distance_Type.Cosine_Distance,
                 b=1.0, h=0.1, feature_weight=0.1, device=None):
        super(ContextualLoss, self).__init__()
        listen_list = []
        self.layers_weights = {}
        try:
            listen_list = layers_weights.keys()
            self.layers_weights = layers_weights
        except:
            pass
        self.vgg_pred = VGG_Model(listen_list=listen_list)
        # self.vgg_gt = VGG_Model(listen_list=listen_list)
        # if cuda:
        #     self.vgg_pred = nn.DataParallel(self.vgg_pred.cuda())
            # self.vgg_gt = nn.DataParallel(self.vgg_gt.cuda())
        self.crop_quarter = crop_quarter
        self.distanceType = distance_type
        self.max_1d_size = max_1d_size
        self.b = b
        self.h = h
        self.feature_weight = feature_weight
        self.device = device

    def forward(self, images, gt):
        if images.device.type == 'cpu':
            loss = torch.zeros(1)
            vgg_images = self.vgg_pred(images)
            vgg_images = {k: v.clone() for k, v in vgg_images.items()}
            vgg_gt = self.vgg_pred(gt)
        else:
            id_cuda = torch.cuda.current_device()
            loss = torch.zeros(1).cuda(id_cuda)
            vgg_images = self.vgg_pred(images)
            vgg_images = {k: v.clone().cuda(id_cuda) for k, v in vgg_images.items()}
            vgg_gt = self.vgg_pred(gt)
            vgg_gt = {k: v.cuda(id_cuda) for k, v in vgg_gt.items()}
        # print('images', [v.device for k, v in vgg_images.items()])
        # print('gt', [v.device for k, v in vgg_gt.items()])

        for key in self.layers_weights.keys():
            N, C, H, W = vgg_images[key].size()

            if self.crop_quarter:
                vgg_images[key] = self._crop_quarters()

            if H*W > self.max_1d_size**2:
                vgg_images[key] = self._random_pooling(vgg_images[key], output_1d_size=self.max_1d_size)
                vgg_gt[key] = self._random_pooling(vgg_gt[key], output_1d_size=self.max_1d_size)

            loss_t = self.calculate_CX_Loss(vgg_images[key], vgg_gt[key])
            # print(loss_t)
            loss += loss_t * self.layers_weights[key]
            # del vgg_images[key], vgg_gt[key]
        return loss


    @staticmethod
    def _random_sampling(tensor, n, indices):
        N, C, H, W = tensor.size()
        S = H * W
        tensor = tensor.view(N, C, S)
        if indices is None:
            indices = torch.randperm(S)[:n].contiguous().type_as(tensor).long()
            indices = indices.view(1, 1, -1).expand(N, C, -1)
        indices = ContextualLoss._move_to_current_device(indices)

        # print('current_device', torch.cuda.current_device(), tensor.device, indices.device)
        res = torch.gather(tensor, index=indices, dim=-1)
        return res, indices

    @staticmethod
    def _move_to_current_device(tensor):
        if tensor.device.type == 'cuda':
            id = torch.cuda.current_device()
            return tensor.cuda(id)
        return tensor

    @staticmethod
    def _random_pooling(feats, output_1d_size=100):
        single_input = type(feats) is torch.Tensor

        if single_input:
            feats = [feats]

        N, C, H, W = feats[0].size()
        feats_sample, indices = ContextualLoss._random_sampling(feats[0], output_1d_size**2, None)
        res = [feats_sample]

        for i in range(1, len(feats)):
            feats_sample, _ = ContextualLoss._random_sampling(feats[i], -1, indices)
            res.append(feats_sample)

        res = [feats_sample.view(N, C, output_1d_size, output_1d_size) for feats_sample in res]

        if single_input:
            return res[0]
        return res

    @staticmethod
    def _crop_quarters(feature):
        N, fC, fH, fW = feature.size()
        quarters_list = []
        quarters_list.append(feature[..., 0:round(fH / 2), 0:round(fW / 2)])
        quarters_list.append(feature[..., 0:round(fH / 2), round(fW / 2):])
        quarters_list.append(feature[..., round(fH / 2), 0:round(fW / 2)])
        quarters_list.append(feature[..., round(fH / 2):, round(fW / 2):])

        feature_tensor = torch.cat(quarters_list, dim=0)
        return feature_tensor

    @staticmethod
    def _create_using_L2(I_features, T_features):
        """
        Calculating the distance between each feature of I and T
        :param I_features:
        :param T_features:
        :return: raw_distance: [N, C, H, W, H*W], each element of which is the distance between I and T at each position
        """
        assert I_features.size() == T_features.size()
        N, C, H, W = I_features.size()

        Ivecs = I_features.view(N, C, -1)
        Tvecs = T_features.view(N, C, -1)
        #
        square_I = torch.sum(Ivecs*Ivecs, dim=1, keepdim=False)
        square_T = torch.sum(Tvecs*Tvecs, dim=1, keepdim=False)
        # raw_distance
        raw_distance = []
        for i in range(N):
            Ivec, Tvec, s_I, s_T = Ivecs[i, ...], Tvecs[i, ...], square_I[i, ...], square_T[i, ...]
            # matrix multiplication
            AB = Ivec.permute(1, 0) @ Tvec
            dist = s_I.view(-1, 1) + s_T.view(1, -1) - 2*AB

            raw_distance.append(dist.view(1, H, W, H*W))
        raw_distance = torch.cat(raw_distance, dim=0)
        raw_distance = torch.clamp(raw_distance, 0.0)
        return raw_distance

    @staticmethod
    def _create_using_L1(I_features, T_features):
        assert I_features.size() == T_features.size()
        N, C, H, W = I_features.size()

        Ivecs = I_features.view(N, C, -1)
        Tvecs = T_features.view(N, C, -1)

        raw_distance = []
        for i in range(N):
            Ivec, Tvec = Ivecs[i, ...], Tvecs[i, ...]
            dist = torch.sum(
                torch.abs(Ivec.view(C, -1, 1) - Tvec.view(C, 1, -1)), dim=0, keepdim=False
            )
            raw_distance.append(dist.view(1, H, W, H*W))
        raw_distance = torch.cat(raw_distance, dim=0)
        return raw_distance

    @staticmethod
    def _centered_by_T(I, T):
        mean_T = T.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        # print(I.device, T.device, mean_T.device)
        return I-mean_T, T-mean_T

    @staticmethod
    def _normalized_L2_channelwise(tensor):
        norms = tensor.norm(p=2, dim=1, keepdim=True)
        return tensor / norms

    @staticmethod
    def _create_using_dotP(I_features, T_features):
        assert I_features.size() == T_features.size()
        I_features, T_features = ContextualLoss._centered_by_T(I_features, T_features)
        I_features = ContextualLoss._normalized_L2_channelwise(I_features)
        T_features = ContextualLoss._normalized_L2_channelwise(T_features)

        N, C, H, W = I_features.size()
        cosine_dist = []
        for i in range(N):
            T_features_i = T_features[i].view(1, 1, C, H*W).permute(3, 2, 0, 1).contiguous()
            I_features_i = I_features[i].unsqueeze(0)
            dist = F.conv2d(I_features_i, T_features_i).permute(0, 2, 3, 1).contiguous()
            cosine_dist.append(dist)
        cosine_dist = torch.cat(cosine_dist, dim=0)
        cosine_dist = (1 - cosine_dist) / 2
        cosine_dist = cosine_dist.clamp(min=0.0)
        return cosine_dist

    def _compute_meshgrid(self, shape):
        N, C, H, W = shape
        rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
        cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)

        feature_grid = torch.meshgrid(rows, cols)
        feature_grid = torch.stack(feature_grid).unsqueeze(0)
        feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0).to(self.device)

        return feature_grid

    @staticmethod
    def _calculate_relative_distance(raw_distance, epsilon=1e-5):
        """
        Normalizing the distances first as Eq. (2) in paper
        :param raw_distance:
        :param epsilon:
        :return:
        """
        div = torch.min(raw_distance, dim=-1, keepdim=True)[0]
        relative_dist = raw_distance / (div + epsilon)
        return relative_dist

    def compute_meshgrid(self, shape):
        N, C, H, W = shape
        rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
        cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)

        feature_grid = torch.meshgrid(rows, cols)
        feature_grid = torch.stack(feature_grid).unsqueeze(0)
        feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)

        return feature_grid

    def calculate_CX_Loss(self, I_features, T_features):
        I_features = ContextualLoss._move_to_current_device(I_features)
        T_features = ContextualLoss._move_to_current_device(T_features)

        if torch.sum(torch.isnan(I_features)) == torch.numel(I_features) or torch.sum(torch.isinf(I_features)) == torch.numel(I_features):
            print(I_features)
            raise ValueError('NaN or Inf in I_features')
        if torch.sum(torch.isnan(T_features)) == torch.numel(T_features) or torch.sum(
                torch.isinf(T_features)) == torch.numel(T_features):
            print(T_features)
            raise ValueError('NaN or Inf in T_features')

        if self.distanceType == Distance_Type.L1_Distance:
            raw_distance = ContextualLoss._create_using_L1(I_features, T_features)
        elif self.distanceType == Distance_Type.L2_Distance:
            raw_distance = ContextualLoss._create_using_L2(I_features, T_features)
        else:
            raw_distance = ContextualLoss._create_using_dotP(I_features, T_features)

        if torch.sum(torch.isnan(raw_distance)) == torch.numel(raw_distance) or torch.sum(
                torch.isinf(raw_distance)) == torch.numel(raw_distance):
            print(raw_distance)
            raise ValueError('NaN or Inf in raw_distance')

        relative_distance = ContextualLoss._calculate_relative_distance(raw_distance)
        if torch.sum(torch.isnan(relative_distance)) == torch.numel(relative_distance) or torch.sum(
                torch.isinf(relative_distance)) == torch.numel(relative_distance):
            print(relative_distance)
            raise ValueError('NaN or Inf in relative_distance')
        del raw_distance

        exp_distance = torch.exp((self.b - relative_distance) / self.h)
        if torch.sum(torch.isnan(exp_distance)) == torch.numel(exp_distance) or torch.sum(
                torch.isinf(exp_distance)) == torch.numel(exp_distance):
            print(exp_distance)
            raise ValueError('NaN or Inf in exp_distance')
        del relative_distance
        # Similarity
        contextual_sim = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True)
        if torch.sum(torch.isnan(contextual_sim)) == torch.numel(contextual_sim) or torch.sum(
                torch.isinf(contextual_sim)) == torch.numel(contextual_sim):
            print(contextual_sim)
            raise ValueError('NaN or Inf in contextual_sim')
        del exp_distance
        max_gt_sim = torch.max(torch.max(contextual_sim, dim=1)[0], dim=1)[0]
        del contextual_sim
        CS = torch.mean(max_gt_sim, dim=1)
        CX_loss = torch.mean(-torch.log(CS))
        if torch.isnan(CX_loss):
            raise ValueError('NaN in computing CX_loss')
        return CX_loss

    def calculate_bilateral_CX_Loss(self, I_features, T_features):

        grid = self.compute_meshgrid(I_features.shape).to(T_features.device)
        raw_distance = ContextualLoss._create_using_L2(grid, grid)
        dist_tilde = ContextualLoss._calculate_relative_distance(raw_distance)
        exp_distance = torch.exp((self.b - dist_tilde) / self.h)
        cx_sp = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True)

        if self.distanceType == 'l1':
            raw_distance = ContextualLoss._create_using_L1(I_features, T_features)
        elif self.distanceType == 'l2':
            raw_distance = ContextualLoss._create_using_L2(I_features, T_features)
        else: # self.distanceType == 'cosine':
            raw_distance = ContextualLoss._create_using_dotP(I_features, T_features)
        dist_tilde = ContextualLoss._calculate_relative_distance(raw_distance)
        exp_distance = torch.exp((self.b - dist_tilde) / self.h) # Eq(3)
        cx_feat = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True) # Eq(4)

        # combined loss
        cx_combine = (1. - self.feature_weight) * cx_feat + self.feature_weight * cx_sp
        # k_max_NC = torch.max(torch.max(cx_combine, dim=1)[0], dim=1)[0]
        k_max_NC, _ = torch.max(cx_combine, dim=2, keepdim=True)
        cx = torch.mean(k_max_NC, dim=1)
        cx_loss = torch.mean(-torch.log(cx + 1e-5))
        return cx_loss

# import torch
# import torch.nn as nn
# from .vgg_model import VGG19
# import torch.nn.functional as F
# LOSS_TYPES = ['cosine']


# def contextual_loss(x=torch.Tensor,
#                     y=torch.Tensor,
#                     band_width= 0.5,
#                     loss_type= 'cosine'):
#     """
#     Computes contextual loss between x and y.
#     The most of this code is copied from
#         https://gist.github.com/yunjey/3105146c736f9c1055463c33b4c989da.
#     Parameters
#     ---
#     x : torch.Tensor
#         features of shape (N, C, H, W).
#     y : torch.Tensor
#         features of shape (N, C, H, W).
#     band_width : float, optional
#         a band-width parameter used to convert distance to similarity.
#         in the paper, this is described as :math:`h`.
#     loss_type : str, optional
#         a loss type to measure the distance between features.
#     Returns
#     ---
#     cx_loss : torch.Tensor
#         contextual loss between x and y (Eq (1) in the paper)
#     """
#     #print('band_width:',band_width)
#     #assert x.size() == y.size(), 'input tensor must have the same size.'
#     assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'

#     N, C, H, W = x.size()

#     if loss_type == 'cosine':
#         dist_raw = compute_cosine_distance(x, y)

#     dist_tilde = compute_relative_distance(dist_raw)
#     cx_ = compute_cx(dist_tilde, band_width)

#     r_m = torch.max(cx_, dim=1, keepdim=True)
#     c = torch.gather(torch.exp((1 - dist_raw) / 0.5), 1, r_m[1])
#     rank = torch.distributed.get_rank()
#     cx = torch.sum(torch.squeeze(r_m[0]*c,1), dim=1) / torch.sum(torch.squeeze(c,1), dim=1)
#     cx_loss = torch.mean(-torch.log(cx + 1e-5)) 

#     c = c.view(N, 1, y.shape[2], y.shape[3])
#     return cx_loss, c

# def compute_meshgrid(shape):
#     N, C, H, W = shape
#     rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
#     cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)

#     feature_grid = torch.meshgrid(rows, cols)
#     feature_grid = torch.stack(feature_grid).unsqueeze(0)
#     feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)

#     return feature_grid

# def contextual_bilateral_loss(x=torch.Tensor,
#                     y=torch.Tensor,
#                     weight_sp= 0.1,
#                     band_width= 0.5,
#                     loss_type= 'cosine'):

#     assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'

#     N, C, H, W = x.size()

#     grid = compute_meshgrid(x.shape).to(x.device)
#     dist_raw = compute_l2_distance(grid, grid)
#     dist_tilde = compute_relative_distance(dist_raw)
#     cx_sp = compute_cx(dist_tilde, band_width)

#     if loss_type == 'cosine':
#         dist_raw = compute_cosine_distance(x, y)
#     elif loss_type == 'L2':
#         dist_raw = compute_l2_distance(x, y)
 
#     dist_tilde = compute_relative_distance(dist_raw)
#     cx_ = compute_cx(dist_tilde, band_width)

#     cx_ = (1. - weight_sp) * cx_ + weight_sp * cx_sp

#     r_m = torch.max(cx_, dim=1, keepdim=True)
#     c = torch.gather(torch.exp((1 - dist_raw) / band_width), 1, r_m[1])
#     # print('\n\n', dist_tilde.min(), dist_tilde.mean(), dist_tilde.max(), '\n\n')
#     rank = torch.distributed.get_rank()
#     cx = torch.sum(torch.squeeze(r_m[0]*c,1), dim=1) / torch.sum(torch.squeeze(c,1), dim=1)
#     cx_loss = torch.mean(-torch.log(cx + 1e-5)) 

#     c = c.view(N, 1, y.shape[2], y.shape[3])
#     return cx_loss, c

# def compute_cx(dist_tilde, band_width):
#     w = torch.exp((1 - dist_tilde) / band_width)  # Eq(3)
#     cx = w / (torch.sum(w, dim=2, keepdim=True) + 1e-5)  # Eq(4)
#     return cx

# def compute_relative_distance(dist_raw):
#     dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
#     dist_tilde = dist_raw / (dist_min + 1e-5)
#     return dist_tilde

# def compute_l2_distance(x, y):
#     N, C, H, W = x.size()
#     x_vec = x.view(N, C, -1)
#     y_vec = y.view(N, C, -1)
#     x_s = torch.sum(x_vec ** 2, dim=1, keepdim=True)
#     y_s = torch.sum(y_vec ** 2, dim=1, keepdim=True)
#     A = y_vec.transpose(1, 2) @ x_vec
#     # print(x.shape, y_s.shape, A.shape, x_s.shape)
#     dist = y_s - 2 * A + x_s
#     dist = dist.transpose(1, 2).reshape(N, H*W, H*W)
#     dist = dist.clamp(min=0.)
#     return dist


# def compute_cosine_distance(x, y):
#     # mean shifting by channel-wise mean of `y`.
#     y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
#     x_mu = x.mean(dim=(0, 2, 3), keepdim=True)
#     x_centered = x - x_mu
#     y_centered = y - y_mu

#     # L2 normalization
#     x_normalized = F.normalize(x_centered, p=2, dim=1)
#     y_normalized = F.normalize(y_centered, p=2, dim=1)

#     # channel-wise vectorization
#     N, C, *_ = x.size()
#     x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
#     y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

#     # # consine similarity
#     # cosine_sim = torch.bmm(x_normalized.transpose(1, 2),
#     #                        y_normalized)  # (N, H*W, H*W)

#     # # convert to distance
#     # dist = 1 - cosine_sim
#     # dist = torch.clamp(dist, min=0)

#     dist = torch.clamp(1 - torch.bmm(x_normalized.transpose(1, 2), y_normalized), min=0)


#     return dist

# @LOSS_REGISTRY.register()
# class ContextualLoss(nn.Module):
#     """
#     Creates a criterion that measures the contextual loss.
#     Parameters
#     ---
#     band_width : int, optional
#         a band_width parameter described as :math:`h` in the paper.
#     use_vgg : bool, optional
#         if you want to use VGG feature, set this `True`.
#     vgg_layer : str, optional
#         intermidiate layer name for VGG feature.
#         Now we support layer names:
#             `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
#     """

#     def __init__(self,
#                  band_width = 0.5,
#                  loss_type = 'cosine',
#                  is_CoBi = False,
#                  use_vgg = True,
#                  vgg_layer = 'relu3_4'):
#         super(ContextualLoss, self).__init__()


#         self.band_width = band_width
#         self.is_CoBi = is_CoBi

#         if use_vgg:
#             # print('use_vgg:',use_vgg)
#             self.vgg_model = VGG19()
#             self.vgg_layer = vgg_layer
#             self.register_buffer(
#                 name='vgg_mean',
#                 tensor=torch.tensor(
#                     [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False)
#             )
#             self.register_buffer(
#                 name='vgg_std',
#                 tensor=torch.tensor(
#                     [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False)
#             )

#     def forward(self, x, y):
#         if hasattr(self, 'vgg_model'):
#             assert x.shape[1] == 3 and y.shape[1] == 3,\
#                 'VGG model takes 3 channel images.'
#             # normalization
#             x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
#             y = y.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())

#             # picking up vgg feature maps
#             x = getattr(self.vgg_model(x), self.vgg_layer)
#             y = getattr(self.vgg_model(y), self.vgg_layer)


#         if self.is_CoBi:
#             return contextual_bilateral_loss(x, y, band_width=self.band_width)
#         else:
#             return contextual_loss(x, y, band_width=self.band_width)