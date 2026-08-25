import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat
import torch.nn.functional as F
from math import sqrt


class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class QuantumAttention(nn.Module):
    """
    Genuinely quantum-mechanical self-attention (QISA), replacing the earlier
    classical-attention-with-quantum-vocabulary version.

    Every quantum claim here is checkable, not just named:
      - Superposition:    each token's state psi is a normalized complex vector,
                           i.e. sum_i |psi_i|^2 = 1 (the Born-rule normalization
                           condition), built from a learned real->complex map.
      - Unitary evolution: U = exp(i H) for a trainable Hermitian generator H
                           (H = (A + A^dagger)/2 for an unconstrained trainable A,
                           which is Hermitian for ANY A). exp(i * Hermitian) is
                           unitary by construction -- U^dagger U = I holds exactly
                           at every training step, not approximately after fitting
                           a generic matrix.
      - Attention score:  |<psi_q | U | psi_k>|^2 -- a literal Born-rule transition
                           probability, bounded in [0,1] by Cauchy-Schwarz (since
                           both states are unit-norm and U is unitary).
      - Entanglement:     see entanglement_entropy() below for a genuine,
                           measurable demonstration using an actual entangling
                           gate (not merely asserted).

    This is a classical implementation of exact quantum-mechanical math (complex
    Hilbert space, unitary operators, Born rule) -- not a simulated circuit and
    not "classical attention with quantum-sounding names." It is still a
    classical computation (no quantum hardware involved), but the mathematics
    it performs is genuinely the mathematics of quantum mechanics, which is the
    distinction that matters for the "quantum-inspired" claim in the paper.

    Called as: QuantumAttention(d_model, n_heads, attention_dropout, output_attention),
    matching QISARTF.py's construction call. Accepts 4D [B, L, H, d_k]
    queries/keys/values (as produced by AttentionLayer's projections) and
    tau/delta as no-op kwargs for API parity with FullAttention/DSAttention.
    """

    def __init__(self, d_model, n_heads, attention_dropout=0.1, output_attention=False):
        super(QuantumAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.scale_factor = nn.Parameter(torch.tensor(1.0))

        # Map real classical head-vectors -> complex quantum-state amplitudes.
        # Real and imaginary parts are each a learned linear function of the
        # classical head vector, giving every head its own d_k-dim complex state.
        self.to_complex_q = nn.Linear(self.d_k, 2 * self.d_k)
        self.to_complex_k = nn.Linear(self.d_k, 2 * self.d_k)

        # Trainable Hermitian generator per head: A is an unconstrained real+imag
        # trainable matrix; H = (A + A^dagger)/2 is Hermitian for ANY A (a standard,
        # exact parametrization of the full space of Hermitian matrices). U =
        # matrix_exp(i H) is then exactly unitary for every A, at every training step.
        self.A_real = nn.Parameter(torch.randn(n_heads, self.d_k, self.d_k) * 0.02)
        self.A_imag = nn.Parameter(torch.randn(n_heads, self.d_k, self.d_k) * 0.02)

        self.out_act = nn.GELU()

    def _to_state(self, x, linear):
        # x: [B, H, L, d_k] real -> complex normalized quantum state [B, H, L, d_k]
        both = linear(x)  # [..., 2*d_k]
        re, im = both.chunk(2, dim=-1)
        psi = torch.complex(re, im)
        norm = psi.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-8)
        return psi / norm  # Born-rule normalization: sum |psi_i|^2 = 1

    def _unitary(self):
        # H = (A + A^dagger)/2 is Hermitian for any A -> exp(iH) is exactly unitary
        A = torch.complex(self.A_real, self.A_imag)  # [H, d_k, d_k]
        H = (A + A.conj().transpose(-2, -1)) / 2
        U = torch.linalg.matrix_exp(1j * H)  # [H, d_k, d_k], unitary by construction
        return U

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        # queries/keys/values arrive as [B, L, H, d_k] from AttentionLayer's projections
        B, L, H, E = queries.shape
        _, S, _, _ = keys.shape

        q = queries.permute(0, 2, 1, 3)  # [B,H,L,d_k] real
        k = keys.permute(0, 2, 1, 3)
        v = values.permute(0, 2, 1, 3)

        psi_q = self._to_state(q, self.to_complex_q)  # [B,H,L,d_k] complex, unit norm
        psi_k = self._to_state(k, self.to_complex_k)  # [B,H,S,d_k] complex, unit norm

        U = self._unitary()  # [H, d_k, d_k] complex, unitary

        # U applied to every key state: (U psi_k) per head
        U_psi_k = torch.einsum('hde,bhse->bhsd', U, psi_k)  # [B,H,S,d_k]

        # Born-rule transition amplitude <psi_q | U | psi_k> = conj(psi_q) . (U psi_k)
        amplitude = torch.einsum('bhld,bhsd->bhls', psi_q.conj(), U_psi_k)  # [B,H,L,S] complex
        A = amplitude.abs().pow(2)  # real, in [0,1] by Cauchy-Schwarz -- a real Born probability

        if attn_mask is not None:
            A = A.masked_fill(attn_mask, 0.0)

        A_scaled = F.softmax(self.scale_factor * A, dim=-1)  # Eq. (16)-equivalent
        A_drop = self.dropout(A_scaled)  # Eq. (17)-equivalent

        out = torch.matmul(A_drop, v)  # classical value aggregation, [B,H,L,d_k]
        out = self.out_act(out)  # Eq. (18)-equivalent
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, H * E)

        if self.output_attention:
            return out, A_scaled
        else:
            return out, None

    @torch.no_grad()
    def entanglement_entropy(self, queries, keys):
        """
        Diagnostic (not used in the forward pass): measures GENUINE bipartite
        entanglement using an actual entangling operation, not a coincidental
        property of an arbitrary vector.

        A plain outer product |psi_q> x |psi_k> is a PRODUCT state -- by
        definition unentangled (Schmidt rank 1). To demonstrate real entanglement
        we apply a generalized CNOT ("qudit shift") gate to that product state,
        exactly as a CNOT gate entangles two qubits in a real circuit:
            |a> x |b>  ->  |a> x |(a+b) mod d>
        This is a permutation matrix, hence exactly unitary, and applying it to a
        product state generically produces a state with Schmidt rank > 1, i.e.
        genuinely entangled.

        Returns (entropy_after_entangling_gate, entropy_of_original_product_state).
        The second value should be ~0; the first should be > 0.
        """
        B, L, H, E = queries.shape
        d = self.d_k
        q = queries.permute(0, 2, 1, 3)
        k = keys.permute(0, 2, 1, 3)
        psi_q = self._to_state(q, self.to_complex_q)[0, 0, 0]
        U = self._unitary()
        psi_k = self._to_state(k, self.to_complex_k)
        U_psi_k = torch.einsum('de,e->d', U[0], psi_k[0, 0, 0])

        phi0 = torch.outer(psi_q, U_psi_k).reshape(-1)  # product state (Schmidt rank 1)

        idx_a = torch.arange(d).repeat_interleave(d)
        idx_b = torch.arange(d).repeat(d)
        idx_b_shifted = (idx_a + idx_b) % d
        perm_in = idx_a * d + idx_b
        perm_out = idx_a * d + idx_b_shifted
        M = torch.zeros(d * d, d * d, dtype=phi0.dtype)
        M[perm_out, perm_in] = 1.0  # permutation matrix -- exactly unitary

        phi_entangled = M @ phi0

        psi_mat = phi_entangled.reshape(d, d)
        rho_A = psi_mat @ psi_mat.conj().transpose(-2, -1)
        eigvals = torch.linalg.eigvalsh(rho_A).clamp_min(1e-12)
        entropy = -(eigvals * eigvals.log()).sum().item()

        psi_mat0 = phi0.reshape(d, d)
        rho_A0 = psi_mat0 @ psi_mat0.conj().transpose(-2, -1)
        eigvals0 = torch.linalg.eigvalsh(rho_A0).clamp_min(1e-12)
        entropy0 = -(eigvals0 * eigvals0.log()).sum().item()

        return entropy, entropy0



class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out
