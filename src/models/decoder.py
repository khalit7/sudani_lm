from fsspec.core import conf
import torch
from torch.mps import is_available
import torch.nn as nn
from torch.profiler import ProfilerActivity,profile,record_function

class PositionalEmbedding(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.d_model     = config["d_model"]
        self.max_seq_len = config["max_seq_len"]

        if self.d_model%2 != 0:
            raise Exception("d_model should have an even value")

        pos_encoding = self._get_pos_encoding() # (max_seq_len,d_model)

        self.register_buffer("pos_encoding",pos_encoding)


    def forward(self,x):
        # x has shape (batch_size,seq_len,d_model)
        _,seq_len,_ = x.shape

        return x + self.pos_encoding[0:seq_len,:].unsqueeze(0)

    def _get_pos_encoding(self):
        
        pos_encoding = torch.empty(self.max_seq_len,self.d_model)

        pos = torch.arange(start=0,end=self.max_seq_len)
        i   = torch.arange(start=0,end=self.d_model//2)
        i   = 10000**(2*i/self.d_model)

        pos_encoding[:,0::2] = torch.sin( pos.unsqueeze(-1)/i )
        pos_encoding[:,1::2] = torch.cos( pos.unsqueeze(-1)/i )

        return pos_encoding

class MaskedMultiHeadAttn(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.d_model    = config["d_model"]
        self.num_heads  = config["num_heads"]
        self.max_seq_len = config["max_seq_len"]

        if self.d_model%self.num_heads != 0:
            raise Exception("d_model is not divisable by num_heads")

        self.d_k = self.d_model//self.num_heads

        self.k_proj = nn.Linear(self.d_model,self.d_model)
        self.q_proj = nn.Linear(self.d_model,self.d_model)
        self.v_proj = nn.Linear(self.d_model,self.d_model)

        self.out_proj = nn.Linear(self.d_model,self.d_model)

        causal_mask = torch.tril(torch.ones(self.max_seq_len,self.max_seq_len),diagonal=0).bool() # TODO: save space by converting this to bool
        self.register_buffer("causal_mask",causal_mask)

    def forward(self,x,attention_mask):
        # x and has the shape:          (batch_size,seq_len,d_model)
        # attention mask has the shape: (batch_size,seq_len)
        batch_size,seq_len,_ = x.shape
        
        K = self.k_proj(x)                  #(batch_size,seq_len,d_model) 
        Q = self.q_proj(x)                  #(batch_size,seq_len,d_model) 
        V = self.v_proj(x)                  #(batch_size,seq_len,d_model) 

        K = K.view(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)      #(batch_size,num_heads,seq_len,d_k)
        Q = Q.view(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)      #(batch_size,num_heads,seq_len,d_k)
        V = V.view(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)      #(batch_size,num_heads,seq_len,d_k)

        attn_score = K@Q.transpose(-1,-2)                   # (batch_size,num_heads,seq_len,seq_len)
        final_mask = attention_mask.unsqueeze(1).unsqueeze(1).bool() & self.causal_mask[0:seq_len,0:seq_len].unsqueeze(0).unsqueeze(0)
        attn_score = attn_score.masked_fill(torch.logical_not(final_mask),value=float("-inf"))
        attn_score = torch.nn.functional.softmax(attn_score,dim=-1)

        x = attn_score@V            #(batch_size,num_heads,seq_len,d_k)

        x = x.transpose(1,2).contiguous().view(batch_size,seq_len,self.d_model)     # (batch_size,seq_len,d_model)
        x = self.out_proj(x)

        return x





class DecoderLayer(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.d_model = config["d_model"]
        
        self.masked_multihead_attn = MaskedMultiHeadAttn(config)
        self.norm1 = nn.RMSNorm(self.d_model)
        self.mlp   = nn.Sequential(
                nn.Linear(self.d_model,4*self.d_model),
                nn.GELU(),
                nn.Linear(4*self.d_model,self.d_model)
                )
        self.norm2 = nn.RMSNorm(self.d_model)

    def forward(self,x,attention_mask):
        
        attn_output = self.masked_multihead_attn(x,attention_mask)
        x = self.norm1(attn_output+x)

        mlp_output = self.mlp(x)
        x = self.norm2(mlp_output + x)

        return x

class Decoder(nn.Module):
    def __init__(self,config ) -> None:
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.d_model    = config["d_model"]
        self.num_layers = config["num_layers"]
        self.token_embedding = nn.Embedding(self.vocab_size,self.d_model)
        self.pos_embedding   = PositionalEmbedding(config)
        self.decoder_layers  = nn.ModuleList( DecoderLayer(config) for _ in range(self.num_layers) ) # TODO: implement
        self.head            = nn.Linear(self.d_model,self.vocab_size)

    def forward(self,input_ids,attention_mask):
        # input_ids has shape        (batch_size,seq_len)
        # atteniton_mask has shape   (batch_size,seq_len)
        x = self.token_embedding(input_ids)     # (batch_size,seq_len,d_model)
        x = self.pos_embedding(x)               # (batch_size,seq_len,d_model)
        for layer in self.decoder_layers:
            x = layer(x,attention_mask)         # (batch_size,seq_len,d_model)
        x = self.head(x)                        # (batch_size,seq_len,vocab_size)

        return x.view(-1,self.vocab_size)
    
    def get_model_stats(self,verbose=True):
        param_size = 0
        num_params = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
            num_params += param.nelement()
        buffer_size = 0
        num_buffers = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            num_buffers += buffer.nelement()

        if verbose:
            print("-"*20)
            print("number of parameters   : ", num_params)
            print("number of buffers      :", num_buffers)
            print("total param size in MB :", param_size/1024**2)
            print("total buffer size in MB:", buffer_size/1024**2)
            print("-"*20)

        return {
                "num_params":num_params,
                "num_buffers":num_buffers,
                "param size (MB)": param_size/1024**2,
                "buffer size (MB)": buffer_size/1024**2
                }

    def profile_model(self,dummy_input):
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        self.to(device)
        dummy_input = {k:v.to(device) for k,v in dummy_input.items()}
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],profile_memory=True,record_shapes=True) as prof:
            self(**dummy_input)
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))


