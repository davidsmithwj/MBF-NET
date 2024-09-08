import torch.nn.functional as F
import torch.nn as nn
import torch

class CrossAttention(nn.Module):
    def __init__(self,
                 d_model,#表示输入的特征向量的维度或者模型的隐藏层大小
                 nhead,#表示多头注意力机制（Multi-Head Attention）中的头数。
                 dim_feedforward=512,#全连接层隐藏层的大小
                 dropout=0.0):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #一个nn.MultiheadAttention模块，用于实现多头注意力机制。它接受输入的维度为d_model，
        # 头的数量为nhead，并可选择使用dropout进行随机失活。
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        #两个线性层，分别是从输入特征向量d_model到隐藏层维度dim_feedforward的映射
        # 并将隐藏层特征向量映射回d_model大小的向量。这两个线性层用于在注意力机制之后进行特征变换和维度变换。
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        #用于规范化输入特征向量或其他层的输出。这些层归一化层有助于加速模型的训练，
        # 并提高模型的泛化能力。

        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        #两个 dropout 层，用于在特征变换和层之间应用随机失活。
        # 它们接受输入特征向量，并以一定比例随机将元素置为零，以防止过拟合。
        
        self.activation = nn.LeakyReLU()

    
    def forward(self, tgt, src):
        "tgt shape: Batch_size, C, H, W "
        "src shape: Batch_size, 1, C    "
        
        B, C, h, w = tgt.shape
        tgt = tgt.view(B, C, h*w).permute(2,0,1)  # shape: L, B, C
        # 将目标重塑为维度为（L, B, C）的形状，其中L是观察序列（或者叫做时间步长）的长度。
        
        src = src.permute(1,0,2)  # shape: Q:1, B, C
        # 将源（src）进行维度变换，将其形状转换为（Q:1, B, C），其中Q是源序列的长度。


        
        fusion_feature = self.cross_attn(query=tgt,
                                         key=src,
                                         value=src)[0]
        tgt = tgt + self.dropout1(fusion_feature)#将目标和注意力计算的输出相加，
        # 并使用dropout1层应用随机失活。


        tgt = self.norm1(tgt)#使用norm1层进行层归一化。
        tgt1 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        #对目标应用线性变换，首先使用linear1进行特征变换，然后使用activation激活函数，
        # 再应用dropout进行随机失活，最后使用linear2进行维度变换。这里得到的tgt1是经过线性层变换后的特征向量。
        tgt = tgt + self.dropout2(tgt1)#将目标和线性变换后的特征向量相加，并使用dropout2层进行随机失活。


        tgt = self.norm2(tgt)
        return tgt.permute(1, 2, 0).view(B, C, h, w)
    #将目标的形状重新变换为（Batch_size, C, H, W）并返回。
    #这个部分的forward方法实现了对目标和源的多头注意力计算、特征变换和维度变换。
    # 它使用层归一化和随机失活来提高模型的表达能力和泛化能力。然后将目标重新变换为与原始形状相同，
    # 并返回结果。

class BoundaryCrossAttention(CrossAttention):
    def __init__(self,
                 d_model,#模型的输入维度。
                 nhead,#头注意力机制中注意力头的数量。
                 BAG_type='2D',#边界自注意力机制的类型，可选值为'2D'或其他。
                 Atrous=True,#布尔值，表示是否使用可调节的空洞卷积。
                 dim_feedforward=512,#全连接层中隐藏层的维度。
                 dropout=0.0):#随机失活的概率。
        super().__init__(d_model, nhead, dim_feedforward, dropout)
        
        #self.BAG = nn.Sequential(
        #    nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
        #    nn.BatchNorm2d(d_model),
        #    nn.ReLU(inplace=False),
        #    nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
        #    nn.BatchNorm2d(d_model),
        #    nn.ReLU(inplace=False),
        #    nn.Conv2d(d_model, 1, kernel_size=1))
        self.BAG_type = BAG_type
        if self.BAG_type == '1D':
            if Atrous:
                self.BAG = BoundaryWiseAttentionGateAtrous1D(d_model)
            else:
                self.BAG = BoundaryWiseAttentionGate1D(d_model)
        elif self.BAG_type == '2D':
            if Atrous:
                self.BAG = BoundaryWiseAttentionGateAtrous2D(d_model)
            else:
                self.BAG = BoundaryWiseAttentionGate2D(d_model)
    
    def forward(self, tgt, src):
        "tgt shape: Batch_size, C, H, W "
        "src shape: Batch_size, 1, C    "
        
        B, C, h, w = tgt.shape
        tgt = tgt.view(B, C, h*w).permute(2,0,1)  # shape: L, B, C
        
        src = src.permute(1,0,2)  # shape: Q:1, B, C
        
        fusion_feature = self.cross_attn(query=tgt,
                                         key=src,
                                         value=src)[0]
        tgt = tgt + self.dropout1(fusion_feature)
        tgt = self.norm1(tgt)
        tgt1 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt1)
        tgt = self.norm2(tgt)

        if self.BAG_type == '1D':
            tgt = tgt.permute(1,2,0)
            tgt, weights = self.BAG(tgt)
            tgt = tgt.view(B, C, h, w).contiguous()
            weights = weights.view(B, 1, h, w)
        elif self.BAG_type == '2D':
            tgt = tgt.permute(1,2,0).view(B, C, h, w)
            tgt, weights = self.BAG(tgt)
            tgt = tgt.contiguous()
            #在这段代码中，contiguous()函数被用于确保张量tgt是连续存储的。在处理连续存储的张量时，
            #某些操作的计算速度更快，因此在需要之前，通过调用contiguous()函数进行内存重排可以提高计算性能。
        return tgt, weights
    
class MultiHeadAttention(nn.Module):
    """
    This class implements a multi head attention module like proposed in:
    https://arxiv.org/abs/2005.12872
    """
    def __init__(self, query_dimension: int = 64, hidden_features: int = 64, number_of_heads: int = 16,
                 dropout: float = 0.0) -> None:
        #设置查询向量的维度，默认值为64。设置隐藏特征数量，默认值为64。设置注意力机制中头的数量，默认值为16。
        """
        Constructor method
        :param query_dimension: (int) Dimension of query tensor
        :param hidden_features: (int) Number of hidden features in detr
        :param number_of_heads: (int) Number of prediction heads
        :param dropout: (float) Dropout factor to be utilized
        """
        # Call super constructor
        super(MultiHeadAttention, self).__init__()
        # Save parameters
        self.hidden_features = hidden_features
        self.number_of_heads = number_of_heads
        self.dropout = dropout
        # Init layer
        self.layer_box_embedding = nn.Linear(in_features=query_dimension, out_features=hidden_features, bias=True)
        # Init convolution layer
        self.layer_image_encoding = nn.Conv2d(in_channels=query_dimension, out_channels=hidden_features,
                                              kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        # Init normalization factor
        self.normalization_factor = torch.tensor(self.hidden_features / self.number_of_heads, dtype=torch.float).sqrt()
        #2
        # Linear
        self.linear = nn.Linear(in_features=number_of_heads, out_features=1)

    def forward(self, input_box_embeddings: torch.Tensor, input_image_encoding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input_box_embeddings: (torch.Tensor) Bounding box embeddings
        :param input_image_encoding: (torch.Tensor) Encoded image of the transformer encoder
        :return: (torch.Tensor) Attention maps of shape (batch size, n, m, height, width)
        """
        # Map box embeddings
        output_box_embeddings = self.layer_box_embedding(input_box_embeddings)
        # Map image features
        output_image_encoding = self.layer_image_encoding(input_image_encoding)
        # Reshape output box embeddings
        output_box_embeddings = output_box_embeddings.view(output_box_embeddings.shape[0],
                                                           output_box_embeddings.shape[1],
                                                           self.number_of_heads,
                                                           self.hidden_features // self.number_of_heads)
        # Reshape output image encoding
        output_image_encoding = output_image_encoding.view(output_image_encoding.shape[0],
                                                           self.number_of_heads,
                                                           self.hidden_features // self.number_of_heads,
                                                           output_image_encoding.shape[-2],
                                                           output_image_encoding.shape[-1])
        # Combine tensors and normalize
        output = torch.einsum("bqnc,bnchw->bqnhw",
                              output_box_embeddings * self.normalization_factor,
                              output_image_encoding)
        # Apply softmax
        output = F.softmax(output.flatten(start_dim=2), dim=-1).view_as(output)

        # Linear: to generate one map
        b, _, _, h, w = output.shape 
        output = torch.sigmoid(self.linear(output.flatten(start_dim=3).permute(0,1,3,2))).view(b,1,h,w)

        # Perform dropout if utilized
        if self.dropout > 0.0:
            output = F.dropout(input=output, p=self.dropout, training=self.training)
#         print("MultiHead Attention",output.shape)
        return output.contiguous()

    
class BoundaryWiseAttentionGateAtrous2D(nn.Module):
    def __init__(self, in_channels, hidden_channels = None):

        super(BoundaryWiseAttentionGateAtrous2D,self).__init__()

        modules = []

        if hidden_channels == None:
            hidden_channels = in_channels // 2

        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)))

        self.convs = nn.ModuleList(modules)
        
        self.conv_out = nn.Conv2d(5 * hidden_channels, 1, 1, bias=False)
    def forward(self, x):
        " x.shape: B, C, H, W "
        " return: feature, weight (B,C,H,W) "
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        weight = torch.sigmoid(self.conv_out(res))
        x = x * weight + x
        return x, weight
    #attention

class BoundaryWiseAttentionGateAtrous1D(nn.Module):
    def __init__(self, in_channels, hidden_channels = None):

        super(BoundaryWiseAttentionGateAtrous1D,self).__init__()

        modules = []

        if hidden_channels == None:
            hidden_channels = in_channels // 2

        modules.append(nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True)))
        modules.append(nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True)))

        self.convs = nn.ModuleList(modules)
        
        self.conv_out = nn.Conv1d(5 * hidden_channels, 1, 1, bias=False)
    def forward(self, x):
        " x.shape: B, C, L "
        " return: feature, weight (B,C,L) "
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        weight = torch.sigmoid(self.conv_out(res))
        x = x * weight + x
        return x, weight

class BoundaryWiseAttentionGate2D(nn.Sequential):
    def __init__(self, in_channels, hidden_channels = None):
        super(BoundaryWiseAttentionGate2D,self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, 1, kernel_size=1))
    def forward(self, x):
        " x.shape: B, C, H, W "
        " return: feature, weight (B,C,H,W) "
        weight = torch.sigmoid(super(BoundaryWiseAttentionGate2D,self).forward(x))
        x = x * weight + x
        return x, weight

class RCAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(128, 128 // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(128 // reduction, 128, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)#2,1,32,32
        avg_out = torch.mean(x, dim=1, keepdim=True)#2,1,32,32
        weight = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))#(2,1,32,32)
        x = x * weight + x
        return x, weight

class BoundaryWiseAttentionGate1D(nn.Sequential):
    def __init__(self, in_channels, hidden_channels = None):
        super(BoundaryWiseAttentionGate1D,self).__init__(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv1d(in_channels, 1, kernel_size=1))
    def forward(self, x):
        " x.shape: B, C, L "
        " return: feature, weight (B,C,L) "
        weight = torch.sigmoid(super(BoundaryWiseAttentionGate1D,self).forward(x))
        x = x * weight + x
        return x, weight