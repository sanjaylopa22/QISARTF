import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, QuantumAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_quantum_attention = getattr(configs, "use_quantum_attention", False)  # Check if QuantumAttention is enabled
        self.down_sampling_window = configs.down_sampling_window
        self.down_sampling_layers = configs.down_sampling_layers
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        QuantumAttention(configs.d_model, configs.n_heads, configs.dropout)
                        if self.use_quantum_attention
                        else FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                           output_attention=configs.output_attention),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # Decoder
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        elif self.task_name in ['anomaly_detection']:
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        elif self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

        # Down-sampling related layers
        self.down_pool = torch.nn.AvgPool1d(self.down_sampling_window)

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if isinstance(param, nn.Linear):
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')  # He initialization
                elif isinstance(param, nn.LayerNorm):
                    nn.init.constant_(param, 1.0)  # Initialize LayerNorm to 1
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        """
        Processes inputs at multiple scales using average pooling.
        """
        x_enc_ori = x_enc
        x_mark_enc_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc)
        x_mark_sampling_list.append(x_mark_enc)

        for _ in range(self.down_sampling_layers):
            x_enc = self.down_pool(x_enc.permute(0, 2, 1))  # Apply pooling on the sequence dimension
            x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
            x_enc_ori = x_enc

            if x_mark_enc is not None:
                x_mark_enc = x_mark_enc[:, ::self.down_sampling_window, :]  # Down-sample time marks
                x_mark_sampling_list.append(x_mark_enc)

        return x_enc_sampling_list, x_mark_sampling_list

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Apply multi-scale processing to the entire input (no decomposition)
        x_enc_sampling_list, x_mark_sampling_list = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        # Use the processed sequences in the model
        enc_out = self.enc_embedding(x_enc_sampling_list[0], x_mark_sampling_list[0])  # Use the finest scale initially
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Apply the decoder to get the forecast output
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :x_enc.shape[2]]
        
        # Return the final output
        dec_out = dec_out + x_enc.mean(1, keepdim=True).detach()  # Revert the normalization if any
        return dec_out

    def anomaly_detection(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = (x_enc - means) / (torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5))

        _, L, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        dec_out = dec_out * torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5) + means
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        elif self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        elif self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        elif self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
