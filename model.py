import math
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # self.register_buffer('pre')
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # y = torch.sum(x * self)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class FeatureAlignment(nn.Module):
    def __init__(self, config):
        super(FeatureAlignment, self).__init__()
        drate = 0.5
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=5),
            nn.BatchNorm1d(128),
            nn.GELU()

        )
        self.inplanes = 128
        self.afr_reduced_cnn_size = 29
        self.AFR = self._make_layer(SEBasicBlock, self.afr_reduced_cnn_size, 1)


    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x1 = x[:, 0:1, :]       #[batch_size, 1, 3000] 
        x2 = x[:, 1:2, :]
        x3 = x[:, 2:3, :]

        x1 = self.features(x1)
        x2 = self.features(x2)
        x3 = self.features(x3)

        x1 = self.AFR(x1).unsqueeze(dim=1)
        x2 = self.AFR(x2).unsqueeze(dim=1)
        x3 = self.AFR(x3).unsqueeze(dim=1)

        x = torch.cat([x1, x2, x3], dim=1)
    
        return x


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model=128, dropout=0.2, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)   # [30, 1]
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # pe:[1, 30, 128]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
    

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.position_single = PositionalEncoding(d_model=config.dim_model, dropout=0.1, max_len=30)
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.dim_model, 
                                                   nhead=config.num_head, 
                                                   dim_feedforward=config.forward_hidden, 
                                                   batch_first=True,
                                                   dropout=config.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder)

        self.drop = nn.Dropout(p=0.5)
        self.layer_norm = nn.LayerNorm(config.dim_model * 3)

        self.position_multi = PositionalEncoding(d_model=config.dim_model * 3, dropout=0.1)
        encoder_layer_multi = nn.TransformerEncoderLayer(d_model=config.dim_model * 3, 
                                                         nhead=config.num_head,
                                                         dim_feedforward=config.forward_hidden, 
                                                         batch_first=True,
                                                         dropout=config.dropout)
        
        self.transformer_encoder_multi = nn.TransformerEncoder(encoder_layer_multi, num_layers=config.num_encoder_multi)


    def forward(self, x):
        x1 = x[:, 0, :, :]
        x2 = x[:, 1, :, :]
        x3 = x[:, 2, :, :]
        x1 = self.position_single(x1)
        x2 = self.position_single(x2)
        x3 = self.position_single(x3)

        x1 = self.transformer_encoder(x1)
        x2 = self.transformer_encoder(x2)
        x3 = self.transformer_encoder(x3)


        x = torch.cat([x1, x2, x3], dim=2)
        x = self.drop(x)
        x = self.layer_norm(x)

        residual = x

        x = self.position_multi(x)
        x = self.transformer_encoder_multi(x)

        x = self.layer_norm(x + residual)

        return x
    

class MultiEncoder(nn.Module):
    def __init__(self, config):
        super(MultiEncoder, self).__init__()

        self.drop = nn.Dropout(p=0.5)
        self.layer_norm = nn.LayerNorm(config.dim_model * 3)

        self.position_multi = PositionalEncoding(d_model=config.dim_model * 3, dropout=0.1, max_len=60)
        encoder_layer_multi = nn.TransformerEncoderLayer(d_model=config.dim_model * 3, 
                                                         nhead=config.num_head,
                                                         dim_feedforward=config.forward_hidden, 
                                                         batch_first=True,
                                                         dropout=config.dropout)
        
        self.transformer_encoder_multi = nn.TransformerEncoder(encoder_layer_multi, num_layers=config.num_encoder_multi)

    def forward(self, x):
        residual = x
        x = self.position_multi(x)
        x = self.transformer_encoder_multi(x)
        x = self.layer_norm(x + residual)       # residual connection

        return x
    

class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        
        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)
        
    def forward(self, x1, x2, mask=None):
        batch_size, seq_len1, seq_len = x1.size()
        seq_len2 = x2.size(1)

        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)
        
        attn = torch.matmul(q1, k2) / self.k_dim**0.5
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear((config.pad_size * 2 ) * config.dim_model * 3, config.fc_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(config.fc_hidden, config.num_classes)
        )

    def forward(self, x1, x2):

        x = x1 + x2

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
    
class CrossFusionSleepNet(nn.Module):
    def __init__(self, config):
        super(CrossFusionSleepNet, self).__init__()

        self.feature_alignment = FeatureAlignment(config)
        self.encoder_1d = Encoder(config)
        self.encoder_tf = Encoder(config)

        self.cross_attention = CrossAttention(in_dim1=config.dim_model * 3, in_dim2=config.dim_model * 3, 
                                              k_dim=config.dim_model * 3 // 4, v_dim=config.dim_model * 3 // 4, num_heads=config.num_head)

        self.multi_encoder_1d = MultiEncoder(config)
        self.multi_encoder_tf = MultiEncoder(config)
        self.feed_forward = FeedForward(config)

    
    def forward(self, x_tf, x_1d):
        
        x_1d = self.feature_alignment(x_1d)     # [bs, 3, 29, 128]
        x_1d = self.encoder_1d(x_1d)            # [bs, 29, 384]
        x_tf = self.encoder_tf(x_tf)            # [bs, 29, 384]

        x = self.cross_attention(x_1d, x_tf)

        x_1d = torch.cat([x_1d, x], dim=1)       # [bs, 58, 384]
        x_tf = torch.cat([x_tf, x], dim=1)       # [bs, 58, 384]

        x_1d = self.multi_encoder_1d(x_1d)      # [bs, 58, 384]
        x_tf = self.multi_encoder_tf(x_tf)      # [bs, 58, 384]

        x = self.feed_forward(x_1d, x_tf)

        return x 
