import matplotlib.pyplot as plt
from torch  import nn
import torch


#EQUATION 1 VISION TRANSFORMER
# x_input = [class_token, image_patch_1, image_patch_2, image_patch_3...] + [class_token_position, image_patch_1_position, image_patch_2_position, image_patch_3_position...]
#Create conv2d layer to turn image into patches
class equation1_layer(nn.Module):
    def __init__(self, input_channel, patch_size, out_channel):
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel,
                   out_channels=out_channel,
                   kernel_size=(patch_size,patch_size),
                   stride=(patch_size,patch_size)),
            nn.Flatten(start_dim=2, end_dim=3)
        )  
    
    def forward(self,x):
        return self.layer_1(x).permute(0,2,1)
    
#EQUATION 2 VISION TRANSFORMER
# x_output_MSA_block = MSA_layer(LN_layer(x_input)) + x_input
#Create multi head self attention layer with layer norm (MSA+LN)
class equation2_layer(nn.Module):
    def __init__(self, embedding_dim = 768, num_heads = 12, atn_dropout = 0.0 ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape= embedding_dim)
        self.mutihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=atn_dropout, batch_first= True)
    
    def forward(self,x):
        x = self.layer_norm(x)
        attn_output, _ = self.mutihead_attention(query = x, key = x, value = x, need_weights = False)
        return attn_output
    
#EQUATION 3 VISION TRANSFORMER
# x_output_MLP_block = MLP_layer(LN_layer(x_output_MSA_block)) + x_output_MSA_block
#Create Multi-layer perceptron block
class equation3_layer(nn.Module):
    def __init__(self, embedding_dim = 768, mlp_size = 3072, dense_dropout:float=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.MLP_block = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.Dropout(p=dense_dropout),
            nn.GELU(),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=dense_dropout)
        )
        
    def forward(self,x):
        x = self.layer_norm(x)
        x = self.MLP_block(x)
        return x

#Transformer Encoder
class transformer_encoder(nn.Module):
    def __init__(self,
                 embedding_dim = 768,
                 num_heads = 12,
                 mlp_size = 3072,
                 atn_dropout = 0.0,
                 dense_dropout:float=0.1):
        super().__init__()
        self.MHA = equation2_layer(embedding_dim = embedding_dim, num_heads = num_heads, atn_dropout = atn_dropout)
        self.MLP = equation3_layer(embedding_dim = embedding_dim, mlp_size = mlp_size, dense_dropout = dense_dropout)
        
    def forward(self,x):
        x = self.MHA(x) + x
        x = self.MLP(x) + x
        return x

#Create the completed Vision Transformer 
class transformer_vision_model(nn.Module):
    def __init__(self,
                 input_channel :int, 
                 out_channel = 768,
                 image_size :int = None,
                 patch_size = 16,
                 embedding_dropout:float=0.1,
                 embedding_dim = 768, 
                 num_heads = 12, 
                 atn_dropout = 0.0,
                 mlp_size = 3072,
                 num_layers = 12, 
                 dense_dropout:float=0.1,
                 num_classes: int = None):
        super().__init__()
        self.class_token = nn.Parameter(data=torch.randn(1, 1, embedding_dim), 
                                        requires_grad=True)
        
        self.position_embedding = nn.Parameter(data=torch.randn(1, int((image_size//patch_size)**2) + 1, embedding_dim), 
                                               requires_grad=True)
        
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        
        self.embedding_layer = equation1_layer(input_channel = input_channel, 
                                               out_channel = out_channel,
                                               patch_size = patch_size)
        
        self.transformer_encoder = nn.Sequential(*[
            transformer_encoder(embedding_dim = embedding_dim, 
                                num_heads = num_heads, 
                                mlp_size = mlp_size, 
                                atn_dropout = atn_dropout, 
                                dense_dropout = dense_dropout) for _ in range(num_layers)]
                                                 )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features= embedding_dim,
                      out_features= num_classes)
        )
        
    def forward(self,x):
        batch_size = x.shape[0]
        
        class_token = self.class_token.expand(batch_size, -1, -1)
        
        x = self.embedding_layer(x)
        
        x = torch.cat((class_token, x), dim=1)
        
        x = self.position_embedding + x
        
        x = self.embedding_dropout(x)
        
        x = self.transformer_encoder(x)
       
        x = self.classifier(x[:, 0])
        return x