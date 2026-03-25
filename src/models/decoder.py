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

        causal_mask = torch.tril(torch.ones(self.max_seq_len,self.max_seq_len),diagonal=0).bool() 
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

class DecoderModel(nn.Module):
    def __init__(self,config ) -> None:
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.num_layers = config["num_layers"]
        self.d_model    = config["d_model"]
        self.token_embedding = nn.Embedding(self.vocab_size,self.d_model)
        self.pos_embedding   = PositionalEmbedding(config)
        self.decoder_layers  = nn.ModuleList( DecoderLayer(config) for _ in range(self.num_layers) ) 

    def forward(self,input_ids,attention_mask):
        # input_ids has shape        (batch_size,seq_len)
        # atteniton_mask has shape   (batch_size,seq_len)
        x = self.token_embedding(input_ids)     # (batch_size,seq_len,d_model)
        x = self.pos_embedding(x)               # (batch_size,seq_len,d_model)
        for layer in self.decoder_layers:
            x = layer(x,attention_mask)         # (batch_size,seq_len,d_model)
        return x
    
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

    def profile_model(self,dummy_input_train,dummy_input_val):
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        self.to(device)
        dummy_input_train = {k:v.to(device) for k,v in dummy_input_train.items()}
        dummy_input_val= {k:v.to(device) for k,v in dummy_input_val.items()}

        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],profile_memory=True,record_shapes=True) as prof:
                self(**dummy_input_val)

        print("eval peak:", torch.cuda.max_memory_allocated()/1024**3)
        torch.cuda.reset_peak_memory_stats()
           
        val_profile = prof.key_averages().table(sort_by="self_cuda_memory_usage")

        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],profile_memory=True,record_shapes=True) as prof:
            self(**dummy_input_train)

        print("train peak:", torch.cuda.max_memory_allocated()/1024**3)
        train_profile = prof.key_averages().table(sort_by="self_cuda_memory_usage")

        return train_profile,val_profile

class DecoderLMHeadModel(DecoderModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.head = nn.Linear(self.d_model,self.vocab_size)

    def forward(self,input_ids,attention_mask,labels=None,chunk_size=10,ignore_index=None):

        hidden_states = super().forward(input_ids,attention_mask) # shape is (batch_size,seq_len,d_model)
        if labels == None: 
            return self.head(hidden_states).view(-1,self.vocab_size) # shape is (batch_size*seq_len,vocab_size)
        else: 
            loss = self.chunked_lm_head(hidden_states.view(-1,self.d_model),labels.view(-1),chunk_size,ignore_index)
            return loss

    def chunked_lm_head(self,hidden_states,labels,chunk_size,ignore_index):
        # TODO: verify this implementation
        # hidden_states has shape (batch_size*seq_len,d_model)
        # labels has shape (batch_size*seq_len)
        num_tokens,_ = hidden_states.shape
        total_tokens = (labels != ignore_index).sum()
        total_loss = 0
        for i in range(0,num_tokens,chunk_size):
            print("running on chunk i",i)
            hidden_states_chunk = hidden_states[i:i+chunk_size,:] # has shape (chunk_size,d_model)
            labels_chunk        = labels[i:i+chunk_size]
            output_chunk = self.head(hidden_states_chunk) # has shape (chunk_size,vocab_size)
            loss_chunk = torch.nn.functional.cross_entropy(output_chunk,labels_chunk,ignore_index=ignore_index,reduction="sum")
            print("loss_chunk : ",loss_chunk)
            loss = loss_chunk/total_tokens
            print("loss : ",loss)
            loss.backward(retain_graph=True)
            total_loss += loss.item()

        return total_loss
