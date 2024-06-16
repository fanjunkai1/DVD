import torch
import torch.nn as nn
import torch.nn.functional as F




def softmax_attention(q, k, v):
    # n x 1(k^2) x nhead x d x h x w
    h, w = q.shape[-2], q.shape[-1]

    q = q.permute(0, 2, 4, 5, 1, 3)  # n x nhead x h x w x 1 x d
    k = k.permute(0, 2, 4, 5, 3, 1)  # n x nhead x h x w x d x k^2
    v = v.permute(0, 2, 4, 5, 1, 3)  # n x nhead x h x w x k^2 x d

    N = q.shape[-1]  # scaled attention
    attn = torch.matmul(q / N ** 0.5, k)
    attn = F.softmax(attn, dim=-1)
    output = torch.matmul(attn, v)

    output = output.permute(0, 4, 1, 5, 2, 3).squeeze(1)
    attn = attn.permute(0, 4, 1, 5, 2, 3).squeeze(1)

    return output, attn

def dotproduct_attention(q, k, v):
    # b x n x d x h x w
    h, w = q.shape[-2], q.shape[-1]

    q = q.flatten(-2).transpose(-2, -1)  # b x n x hw x d
    k = k.flatten(-2)                    # b x n x d x hw
    v = v.flatten(-2).transpose(-2, -1)  # b x n x hw x d

    N = k.shape[-1]
    attn = None
    tmp = torch.matmul(k, v) / N
    output = torch.matmul(q, tmp)

    output = output.transpose(-2, -1)
    output = output.view(*output.shape[:-1], h, w)

    return output, attn

def cosine_attention(q, k, v):
    # n x 1(k^2) x nhead x d x h x w
    h, w = q.shape[-2], q.shape[-1]

    q = q.permute(0, 2, 4, 5, 1, 3)  # n x nhead x h x w x 1 x d
    k = k.permute(0, 2, 4, 5, 3, 1)  # n x nhead x h x w x d x k^2
    v = v.permute(0, 2, 4, 5, 1, 3)  # n x nhead x h x w x k^2 x d

    # print(q.shape, k.shape)
    N = q.shape[-1]  # scaled attention
    attn = cosine_distance(q / N ** 0.5, k)
    attn = F.softmax(attn, dim=-1)
    output = torch.matmul(attn, v)

    output = output.permute(0, 4, 1, 5, 2, 3).squeeze(1)
    attn = attn.permute(0, 4, 1, 5, 2, 3).squeeze(1)

    # print(attn.shape)

    return output, attn

class CosineAttention(nn.Module):
    def __init__(self, feat_dim, n_head, k_size=5, d_k=None, d_v=None):
        super().__init__()
        if d_k is None:
            d_k = feat_dim // n_head
        if d_v is None:
            d_v = feat_dim // n_head

        self.n_head = n_head
        self.k_size = k_size
        self.d_k = d_k
        self.d_v = d_v

        # pre-attention projection
        self.w_qs = nn.Conv2d(feat_dim, n_head * d_k, 1, bias=False)
        self.w_ks = nn.Conv2d(feat_dim, n_head * d_k, 1, bias=False)
        self.w_vs = nn.Conv2d(feat_dim, n_head * d_v, 1, bias=False)

        # after-attention combine heads
        self.fc = nn.Conv2d(n_head * d_v, feat_dim, 1, bias=False)

    def forward(self, q, k, v, flow, attn_type='softmax'):

        # input: n x c x h x w  q, k, v: 1, 64, 256, 256
        # flow: n x 2 x h x w   flow: 1, 2, 256, 256
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        # Pass through the pre-attention projection:
        # n x c x h x w   ---->   n x (nhead*dk) x h x w
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)

        # print(q.shape, k.shape, v.shape) [1, 64, 256, 256]

        n, c, h, w = q.shape

        # ------ Sampling K and V features ---------
        sampling_grid = flow_to_grid(flow, self.k_size)
        # print(sampling_grid.shape) [25, 256, 256, 2]
        # sampled feature
        # n x k^2 x c x h x w
        sample_k_feat = flow_guide_sampler(k, sampling_grid, k_size=self.k_size)
        sample_v_feat = flow_guide_sampler(v, sampling_grid, k_size=self.k_size)

        # print(sample_k_feat.shape, sample_v_feat.shape) [1, 25, 64, 256, 256]

        # Reshape for multi-head attention.
        # n x k^2 x nhead x dk x h x w
        q = q.view(n, 1, n_head, d_k, h, w)
        k = sample_k_feat.view(n, self.k_size**2, n_head, d_k, h, w)
        v = sample_v_feat.view(n, self.k_size**2, n_head, d_v, h, w)


        # -------------- Attention -----------------
        if attn_type == 'softmax':
            # n x 1 x nhead x dk x h x w --> n x nhead x dv x h x w
            q, attn = softmax_attention(q, k, v)
        elif attn_type == 'dot':
            q, attn = dotproduct_attention(q, k, v)
        elif attn_type == 'cosine':
            q, attn = cosine_attention(q, k, v)
        else:
            raise NotImplementedError(f'Unknown attention type {attn_type}')

        # Concatenate all the heads together
        # n x (nhead*dv) x h x w
        q = q.reshape(n, -1, h, w)
        q = self.fc(q)   # n x c x h x w

        # print(q.shape)

        return q, attn
    
def flow_to_grid(flow, k_size=5):
    # flow (Tensor): Tensor with size (n, 2, h, w), normal value.
    # samples = flow + grid + shift
    # n, h, w, _ = flow.size()
    n, _, h, w = flow.size()
    padding = (k_size - 1) // 2

    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid_y = grid_y[None, ...].expand(k_size**2, -1, -1).type_as(flow)
    grid_x = grid_x[None, ...].expand(k_size**2, -1, -1).type_as(flow)

    shift = torch.arange(0, k_size).type_as(flow) - padding
    shift_y, shift_x = torch.meshgrid(shift, shift)
    shift_y = shift_y.reshape(-1, 1, 1).expand(-1, h, w) # k^2, h, w
    shift_x = shift_x.reshape(-1, 1, 1).expand(-1, h, w) # k^2, h, w

    samples_y = grid_y + shift_y # k^2, h, w
    samples_x = grid_x + shift_x # k^2, h, w
    samples_grid = torch.stack((samples_x, samples_y), 3) # k^2, h, w, 2
    samples_grid = samples_grid[None, ...].expand(n, -1, -1, -1, -1) # n, k^2, h, w, 2

    flow = flow.permute(0, 2, 3, 1)[:, None, ...].expand(-1, k_size**2, -1, -1, -1)

    vgrid = samples_grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[..., 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=4).view(-1, h, w, 2)
    # vgrid_scaled.requires_grad = False
    return vgrid_scaled

def flow_guide_sampler(feat, vgrid_scaled, k_size=5, interp_mode='bilinear',
                       padding_mode='zeros', align_corners=True):
    # feat (Tensor): Tensor with size (n, c, h, w).
    # vgrid (Tensor): Tensor with size (nk^2, h, w, 2)
    n, c, h, w = feat.size()
    feat = feat.view(n, 1, c, h, w).expand(-1, k_size**2, -1, -1, -1).reshape(-1, c, h, w)
    sample_feat = F.grid_sample(feat, vgrid_scaled,
                                mode=interp_mode, padding_mode=padding_mode,
                                align_corners=align_corners).view(n, k_size**2, c, h, w)
    return sample_feat


def cosine_distance(x1, x2, eps=1e-8):
    '''
    x1      =  [b, h, n, k]
    x2      =  [b, h, k, m]
    output  =  [b, h, n, m]
    '''
    dots = torch.matmul(x1, x2)
    # print(x1)
    # print(torch.norm(x1, 2, dim = -1))
    scale = torch.einsum('bcghi, bcghj -> bcghij', 
            (torch.norm(x1, 2, dim = -1).clamp(min=eps), torch.norm(x2, 2, dim = -2).clamp(min=eps)))
    
    return (dots / scale)


if __name__ == "__main__":

    x1 = torch.randn((1, 8, 256, 256, 1, 8)).cuda()
    x2 = torch.randn((1, 8, 256, 256, 8, 25)).cuda()

    output = cosine_distance(x1, x2)
    print(output.shape)
