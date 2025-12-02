# Copyright (c) 2025 Minuk Lee. All rights reserved.
# 
# This source code is proprietary and confidential.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# 
# For licensing terms, see the LICENSE file.
# Contact: minuklee@snu.ac.kr
# 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tensordict import TensorDict
from dataclasses import dataclass
from typing import Dict, List, Tuple

# --- 현재 패키지(transformer_solver) 모듈 임포트 ---
from .definitions import (
    FEATURE_DIM, FEATURE_INDEX, SCALAR_PROMPT_FEATURE_DIM,
    NODE_TYPE_PADDING, NODE_TYPE_BATTERY, NODE_TYPE_LOAD, 
    NODE_TYPE_IC, NODE_TYPE_EMPTY
)
from .utils.common import batchify
from .solver_env import PocatEnv, BATTERY_NODE_IDX 


# ---
# 섹션 1: 표준 트랜스포머 빌딩 블록 (효율성)
# ---

class RMSNorm(nn.Module):
    """ Root Mean Square Layer Normalization """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Normalization(nn.Module):
    """ 정규화 레이어 래퍼 (RMSNorm 또는 LayerNorm) """
    def __init__(self, embedding_dim, norm_type='rms', **kwargs):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type == 'rms':
            self.norm = RMSNorm(embedding_dim)
        elif self.norm_type == 'layer':
            self.norm = nn.LayerNorm(embedding_dim)
        else:
            raise NotImplementedError(f"Unknown norm_type: {norm_type}")

    def forward(self, x):
        return self.norm(x)

class ParallelGatedMLP(nn.Module):
    """ SwiGLU FFN (FeedForward) 구현체 """
    def __init__(self, hidden_size: int, **kwargs):
        super().__init__()
        # LLAMA 아키텍처에서 사용하는 FFN 차원 계산
        inner_size = int(2 * hidden_size * 4 / 3)
        multiple_of = 256
        inner_size = multiple_of * ((inner_size + multiple_of - 1) // multiple_of)
        
        self.l1 = nn.Linear(hidden_size, inner_size, bias=False)
        self.l2 = nn.Linear(hidden_size, inner_size, bias=False)
        self.l3 = nn.Linear(inner_size, hidden_size, bias=False)
        self.act = F.silu

    def forward(self, z):
        z1 = self.l1(z)
        z2 = self.l2(z)
        return self.l3(self.act(z1) * z2)

def reshape_by_heads(qkv: torch.Tensor, head_num: int) -> torch.Tensor:
    """ (B, N, H*D) -> (B, H, N, D) """
    batch_s, n = qkv.size(0), qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    return q_reshaped.transpose(1, 2)

def multi_head_attention(q, k, v, attention_mask=None):
    """ 
    표준 Multi-Head Attention 구현.
    (attention_mask가 bool 타입의 (B, ..., N, N)이라고 가정)
    """
    batch_s, head_num, n, key_dim = q.shape
    
    # 1. 스코어 계산
    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / (key_dim ** 0.5)
    
    # 2. 어텐션 마스킹 (마스크가 0/False인 위치를 -inf로)
    if attention_mask is not None:
        if attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1) # (B, N, N) -> (B, 1, N, N)
        
        score_scaled = score_scaled.masked_fill(attention_mask == 0, -1e12)
        
    # 3. Softmax 및 Value 적용
    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)
    
    # 4. (B, H, N, D) -> (B, N, H*D)
    out_transposed = out.transpose(1, 2)
    return out_transposed.contiguous().view(batch_s, n, head_num * key_dim)

class EncoderLayer(nn.Module):
    """ 
    표준 트랜스포머 인코더 레이어 (Post-Normalization)
    """
    def __init__(self, embedding_dim, head_num, qkv_dim, ffd='siglu', **model_params):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        
        self.normalization1 = Normalization(embedding_dim, **model_params)
        
        if ffd == 'siglu':
            self.feed_forward = ParallelGatedMLP(hidden_size=embedding_dim, **model_params)
        else:
            raise NotImplementedError
            
        self.normalization2 = Normalization(embedding_dim, **model_params)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # 1. MHA (Post-Normalization)
        q = reshape_by_heads(self.Wq(x), self.head_num)
        k = reshape_by_heads(self.Wk(x), self.head_num)
        v = reshape_by_heads(self.Wv(x), self.head_num)
        
        mha_out = self.multi_head_combine(multi_head_attention(q, k, v, attention_mask=attention_mask))
        h = self.normalization1(x + mha_out) # Residual + Norm
        
        # 2. FFN (Post-Normalization)
        ffn_out = self.feed_forward(h)
        out = self.normalization2(h + ffn_out) # Residual + Norm
        return out

class PocatDecoderLayer(nn.Module):
    """
    Cross-Attention과 FFN으로 구성된 디코더 레이어
    (Query가 1개이므로 Self-Attention은 생략하고 Cross-Attention에 집중)
    """
    def __init__(self, embedding_dim, head_num, qkv_dim, **model_params):
        super().__init__()
        
        # 1. Cross-Attention (Query는 이전 레이어 출력, Key/Val은 인코더 출력)
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # (Wk, Wv는 인코더 쪽에서 미리 계산된 캐시를 재사용하거나, 여기서 별도 정의 가능)
        # 효율성을 위해 여기서는 인코더의 K, V를 공유(Sharing)하거나 
        # 별도로 투영(Projection)할 수 있습니다. 여기서는 별도 투영을 가정합니다.
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        
        self.norm1 = Normalization(embedding_dim, **model_params)
        self.norm2 = Normalization(embedding_dim, **model_params)
        
        # 2. Feed Forward Network
        self.feed_forward = ParallelGatedMLP(hidden_size=embedding_dim, **model_params)
        
        self.head_num = head_num
        self.qkv_dim = qkv_dim

    def forward(self, x, encoder_out):
        """
        x: (B, 1, D) - 현재 디코더의 Query 상태
        encoder_out: (B, N, D) - 인코더 출력 (Context)
        """
        # --- Cross Attention ---
        # Query: 현재 레이어의 입력 x
        q = reshape_by_heads(self.Wq(x), self.head_num)
        
        # Key, Value: 인코더 출력 (매 레이어마다 새로 계산하여 표현력 증대)
        k = reshape_by_heads(self.Wk(encoder_out), self.head_num)
        v = reshape_by_heads(self.Wv(encoder_out), self.head_num)
        
        mha_out = multi_head_attention(q, k, v)
        mha_out = self.multi_head_combine(mha_out)
        
        h = self.norm1(x + mha_out) # Residual + Norm
        
        # --- FFN ---
        ffn_out = self.feed_forward(h)
        out = self.norm2(h + ffn_out) # Residual + Norm
        
        return out
# ---
# 섹션 2: 디코딩 효율을 위한 캐시
# ---

@dataclass
class PrecomputedCache:
    """
    디코딩 루프에서 반복 계산을 피하기 위해
    인코더의 Key, Value 값을 저장하는 캐시 객체입니다.
    """
    node_embeddings: torch.Tensor
    #glimpse_key: torch.Tensor
    #glimpse_val: torch.Tensor
    logit_key_connect: torch.Tensor # 'Connect' 포인터용 Key
    logit_key_spawn: torch.Tensor   # 'Spawn' 포인터용 Key

    def batchify(self, num_starts: int):
        """ POMO 샘플링을 위해 캐시를 N_starts 배수만큼 복제합니다. """
        return PrecomputedCache(
            batchify(self.node_embeddings, num_starts),
            #batchify(self.glimpse_key, num_starts),
            #batchify(self.glimpse_val, num_starts),
            batchify(self.logit_key_connect, num_starts),
            batchify(self.logit_key_spawn, num_starts),
        )

# ---
# 섹션 3: POCAT 모델 아키텍처
# ---

class PocatPromptNet(nn.Module):
    """
    스칼라/행렬 제약조건을 임베딩하는 프롬프트 네트워크 (N_MAX 대응)
    """
    def __init__(self, embedding_dim: int, N_MAX: int, **kwargs):
        super().__init__()
        self.scalar_net = nn.Sequential(
            nn.Linear(SCALAR_PROMPT_FEATURE_DIM, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 2)
        )
        self.matrix_net = nn.Sequential(
            nn.Linear(N_MAX * N_MAX, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2)
        )
        self.final_processor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )

    def forward(self, scalar_features: torch.Tensor, matrix_features: torch.Tensor) -> torch.Tensor:
        scalar_embedding = self.scalar_net(scalar_features)
        batch_size = matrix_features.shape[0]
        matrix_flat = matrix_features.view(batch_size, -1) # (B, N_MAX*N_MAX)
        matrix_embedding = self.matrix_net(matrix_flat)
        combined_embedding = torch.cat([scalar_embedding, matrix_embedding], dim=-1)
        final_prompt_embedding = self.final_processor(combined_embedding)
        return final_prompt_embedding.unsqueeze(1) # (B, 1, D)


class PocatEncoder(nn.Module):
    """
    Pocat 인코더 (듀얼 어텐션 및 다중 임베딩 주입).
    
    1. 노드 타입(5종)별로 기본 임베딩 적용
    2. 노드 속성/상태(4종)별로 추가 임베딩 주입
    3. 듀얼 어텐션(Sparse/Global) 통과
    """
    def __init__(self, embedding_dim: int, encoder_layer_num: int, **model_params):
        super().__init__()
        
        # 1. 노드 "타입" (5종) 임베딩
        self.embedding_padding = nn.Linear(FEATURE_DIM, embedding_dim)
        self.embedding_battery = nn.Linear(FEATURE_DIM, embedding_dim)
        self.embedding_load = nn.Linear(FEATURE_DIM, embedding_dim)
        self.embedding_ic = nn.Linear(FEATURE_DIM, embedding_dim)
        self.embedding_empty = nn.Linear(FEATURE_DIM, embedding_dim)
        
        # 2. 노드 "속성/상태" (4종) 임베딩 (0 또는 1 값을 인덱스로 사용)
        self.embedding_is_active = nn.Embedding(2, embedding_dim)
        self.embedding_is_template = nn.Embedding(2, embedding_dim)
        self.embedding_can_spawn_into = nn.Embedding(2, embedding_dim)
        self.embedding_rail_type = nn.Embedding(3, embedding_dim) # 0:N/A, 1:Supp, 2:Path

        # 3. 듀얼 어텐션(CaDA) 레이어
        self.sparse_layers = nn.ModuleList([
            EncoderLayer(embedding_dim=embedding_dim, **model_params) 
            for _ in range(encoder_layer_num)
        ])
        self.global_layers = nn.ModuleList([
            EncoderLayer(embedding_dim=embedding_dim, **model_params) 
            for _ in range(encoder_layer_num)
        ])
        self.sparse_fusion = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) 
            for _ in range(encoder_layer_num)
        ])
        self.global_fusion = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) 
            for _ in range(encoder_layer_num - 1)
        ])

    def forward(self, td: TensorDict, prompt_embedding: torch.Tensor) -> torch.Tensor:
        node_features = td['nodes'] # (B, N_MAX, 27)
        batch_size, num_nodes, _ = node_features.shape # num_nodes = N_MAX
        embedding_dim = self.embedding_battery.out_features
        
        node_embeddings = torch.zeros(batch_size, num_nodes, embedding_dim, device=node_features.device)
        
        # --- 1. 타입별 기본 임베딩 적용 ---
        node_type_indices = node_features[..., FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(dim=-1)
        
        masks = {
            NODE_TYPE_PADDING: (node_type_indices == NODE_TYPE_PADDING),
            NODE_TYPE_BATTERY: (node_type_indices == NODE_TYPE_BATTERY),
            NODE_TYPE_LOAD: (node_type_indices == NODE_TYPE_LOAD),
            NODE_TYPE_IC: (node_type_indices == NODE_TYPE_IC),
            NODE_TYPE_EMPTY: (node_type_indices == NODE_TYPE_EMPTY),
        }
        
        if masks[NODE_TYPE_PADDING].any(): node_embeddings[masks[NODE_TYPE_PADDING]] = self.embedding_padding(node_features[masks[NODE_TYPE_PADDING]])
        if masks[NODE_TYPE_BATTERY].any(): node_embeddings[masks[NODE_TYPE_BATTERY]] = self.embedding_battery(node_features[masks[NODE_TYPE_BATTERY]])
        if masks[NODE_TYPE_LOAD].any(): node_embeddings[masks[NODE_TYPE_LOAD]] = self.embedding_load(node_features[masks[NODE_TYPE_LOAD]])
        if masks[NODE_TYPE_IC].any(): node_embeddings[masks[NODE_TYPE_IC]] = self.embedding_ic(node_features[masks[NODE_TYPE_IC]])
        if masks[NODE_TYPE_EMPTY].any(): node_embeddings[masks[NODE_TYPE_EMPTY]] = self.embedding_empty(node_features[masks[NODE_TYPE_EMPTY]])

        # --- 2. 속성/상태 임베딩 주입 (Injection) ---
        active_ids = node_features[..., FEATURE_INDEX["is_active"]].long()
        template_ids = node_features[..., FEATURE_INDEX["is_template"]].long()
        spawn_ids = node_features[..., FEATURE_INDEX["can_spawn_into"]].long()
        rail_ids = node_features[..., FEATURE_INDEX["independent_rail_type"]].round().long().clamp(0, 2)
        
        node_embeddings.add_(self.embedding_is_active(active_ids))
        node_embeddings.add_(self.embedding_is_template(template_ids))
        node_embeddings.add_(self.embedding_can_spawn_into(spawn_ids))
        node_embeddings.add_(self.embedding_rail_type(rail_ids))
        
        # --- 3. 듀얼 어텐션 (CaDA) 실행 ---
        connectivity_mask = td['connectivity_matrix'] # (B, N_MAX, N_MAX)
        attention_mask = td['attention_mask'] # (B, N_MAX, N_MAX)

        global_input = torch.cat((node_embeddings, prompt_embedding), dim=1)
        
        global_attention_mask = torch.zeros(
            batch_size, num_nodes + 1, num_nodes + 1, 
            dtype=torch.bool, device=node_embeddings.device
        )
        global_attention_mask[:, :num_nodes, :num_nodes] = attention_mask
        
        alive_mask_1d = (node_type_indices != NODE_TYPE_PADDING)
        global_attention_mask[:, num_nodes, :num_nodes] = alive_mask_1d
        global_attention_mask[:, :num_nodes, num_nodes] = alive_mask_1d
        global_attention_mask[:, num_nodes, num_nodes] = True
        
        sparse_out, global_out = node_embeddings, global_input
        for i in range(len(self.sparse_layers)):
            sparse_out = self.sparse_layers[i](sparse_out, attention_mask=connectivity_mask)
            global_out = self.global_layers[i](global_out, attention_mask=global_attention_mask)
            
            sparse_out = sparse_out + self.sparse_fusion[i](global_out[:, :num_nodes])
            if i < len(self.global_layers) - 1:
                global_nodes = global_out[:, :num_nodes] + self.global_fusion[i](sparse_out)
                global_out = torch.cat((global_nodes, global_out[:, num_nodes:]), dim=1)  
                
        return global_out[:, :num_nodes] # 프롬프트 임베딩 제외 (B, N_MAX, D)


class PocatDecoder(nn.Module):
    def __init__(self, embedding_dim, head_num, qkv_dim, N_MAX, **model_params):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        self.N_MAX = N_MAX
        
        # config.yaml에서 decoder_layer_num을 가져옵니다 (기본값 1)
        self.layer_num = model_params.get('decoder_layer_num', 1)

        # 1. 초기 컨텍스트 쿼리 생성용 (입력 차원 변환)
        # (embedding_dim + 3 features) -> embedding_dim
        self.input_projector = nn.Linear(embedding_dim + 3, embedding_dim)

        # 2. 디코더 레이어 스택 (ModuleList)
        self.layers = nn.ModuleList([
            PocatDecoderLayer(embedding_dim, head_num, qkv_dim, **model_params)
            for _ in range(self.layer_num)
        ])
        
        # 3. 포인터 네트워크용 Key 생성 (인코더 임베딩을 변환)
        self.Wk_connect_logit = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk_spawn_logit = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # --- 4. 4-Heads (q_vec을 입력으로 받음) ---
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )
        self.type_head = nn.Linear(embedding_dim, 2)
        self.connect_head = nn.Linear(embedding_dim, embedding_dim)
        self.spawn_head = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, td: TensorDict, cache: PrecomputedCache) -> Tuple[torch.Tensor, ...]:
        
        # 1. 초기 쿼리 입력 준비
        avg_current = td["nodes"][..., FEATURE_INDEX["current_out"]].clone().mean(dim=1, keepdim=True)
        unconnected_ratio = td["unconnected_loads_mask"].clone().float().mean(dim=1, keepdim=True)
        step_ratio = td["step_count"].clone().float() / (2 * self.N_MAX)
        state_features = torch.cat([avg_current, unconnected_ratio, step_ratio], dim=1)

        head_idx = td["trajectory_head"].detach().squeeze(-1).clone()
        batch_indices = torch.arange(td.batch_size[0], device=head_idx.device)
        head_emb = cache.node_embeddings[batch_indices, head_idx]
        
        # (B, D+3) -> (B, 1, D)
        query_input = torch.cat([head_emb, state_features], dim=1).unsqueeze(1)
        
        # 초기 q_vec (Projection)
        q_vec = self.input_projector(query_input)

        # 2. 디코더 레이어 순차 통과 (Stacking)
        # q_vec이 각 레이어를 거치며 점점 더 정교한 Context Vector가 됩니다.
        encoder_out = cache.node_embeddings # (B, N, D)
        
        for layer in self.layers:
            q_vec = layer(q_vec, encoder_out)

        # --- 3. 최종 결정 (Heads) ---
        value = self.value_head(q_vec).squeeze(-1)
        logits_action_type = self.type_head(q_vec).squeeze(1)
        
        query_connect = self.connect_head(q_vec) 
        logits_connect_target = torch.matmul(
            query_connect, cache.logit_key_connect
        ).squeeze(1) / (self.embedding_dim ** 0.5)
        
        query_spawn = self.spawn_head(q_vec) 
        logits_spawn_template = torch.matmul(
            query_spawn, cache.logit_key_spawn
        ).squeeze(1) / (self.embedding_dim ** 0.5)

        return logits_action_type, logits_connect_target, logits_spawn_template, value

class PocatModel(nn.Module):
    """
    Pocat V7 (Padding + Lazy Spawn) 메인 모델
    """
    
    def __init__(self, **model_params):
        super().__init__()
        self.logit_clipping = model_params.get('logit_clipping', 10)
        
        # config.yaml에서 N_MAX 주입
        self.N_MAX = model_params['N_MAX']
        # model_params에서 N_MAX를 pop하여 중복 전달 방지
        # (PocatPromptNet과 PocatDecoder는 N_MAX를 명시적 인자로 받음)s
        n_max_value = model_params.pop('N_MAX')
        self.prompt_net = PocatPromptNet(N_MAX=n_max_value, **model_params)
        self.encoder = PocatEncoder(**model_params)
        self.decoder = PocatDecoder(N_MAX=n_max_value, **model_params)

    def _get_masked_probs(self, logits, mask):
        """ 로짓과 마스크를 받아 정규화된 확률 분포를 반환합니다. """
        scores = self.logit_clipping * torch.tanh(logits)
        scores.masked_fill_(~mask, -float('inf'))
        probs = F.softmax(scores, dim=-1)
        return probs  

    def _sample_action(self, logits, mask, decode_type):
        """ 
        로짓과 마스크를 받아 액션(idx)과 로그 확률(log_prob)을 반환합니다.
        (막다른 길 방지 로직 포함)
        """
        scores = self.logit_clipping * torch.tanh(logits)
        scores.masked_fill_(~mask, -float('inf'))
        
        # 모든 액션이 마스킹된 '막다른 길' 상태 방지
        is_stuck = torch.all(scores == -float('inf'), dim=-1)
        scores[is_stuck, 0] = 0.0 # (0번 인덱스(배터리)라도 강제 선택)
        
        log_prob = F.log_softmax(scores, dim=-1)
        probs = log_prob.exp()
        
        if decode_type == 'greedy':
            action = probs.argmax(dim=-1)
        else: # 'sampling'
            action = Categorical(probs=probs).sample()
            
        # 선택된 액션의 로그 확률 반환
        return action, log_prob.gather(1, action.unsqueeze(-1)).squeeze(-1)

    def _combine_log_probs(self, 
                           log_prob_type, action_type, 
                           log_prob_connect, log_prob_spawn):
        """
        Parameterized Action의 로그 확률을 결합합니다.
        logπ(a|s) = logπ(type|s) + logπ(arg|type,s)
        """
        # 'Connect' (0)를 선택한 경우의 로그 확률
        log_prob_if_connect = log_prob_type + log_prob_connect
        # 'Spawn' (1)을 선택한 경우의 로그 확률
        log_prob_if_spawn = log_prob_type + log_prob_spawn
        
        # (B,)
        final_log_prob = torch.where(
            action_type == 0,       # 'Connect'를 선택했으면
            log_prob_if_connect,    # 이 확률을 사용
            log_prob_if_spawn       # 아니면 (Spawn) 이 확률을 사용
        )
        return final_log_prob

    def forward(self, 
                td: TensorDict, 
                env: PocatEnv, # (solver_env.py의 환경 객체)
                decode_type: str = 'greedy', 
                pbar: object = None,
                status_msg: str = "", 
                log_fn=None, log_idx: int = 0, 
                log_mode: str = 'progress',
                return_final_td: bool = False,   # 👈 이 줄 추가
                ) -> Dict[str, torch.Tensor]:
        
        base_desc = pbar.desc.split(' | ')[0] if pbar else ""
        if pbar: pbar.set_description(f"{base_desc} | {status_msg} | ▶ Encoding")
        
        # --- 1. 인코딩 및 캐시 생성 ---
        prompt_embedding = self.prompt_net(td["scalar_prompt_features"], td["matrix_prompt_features"])
        encoded_nodes = self.encoder(td, prompt_embedding) # (B, N_MAX, D)
        
        # 디코더가 사용할 Key/Value 사전 계산
        #glimpse_key = reshape_by_heads(self.decoder.Wk_glimpse(encoded_nodes), self.decoder.head_num)
        #glimpse_val = reshape_by_heads(self.decoder.Wv_glimpse(encoded_nodes), self.decoder.head_num)
        
        # 포인터 헤드별 Key 생성
        logit_key_connect = self.decoder.Wk_connect_logit(encoded_nodes).transpose(1, 2)
        logit_key_spawn = self.decoder.Wk_spawn_logit(encoded_nodes).transpose(1, 2)
        
        cache = PrecomputedCache(
            node_embeddings=encoded_nodes,
            #glimpse_key=glimpse_key,
            #glimpse_val=glimpse_val,
            logit_key_connect=logit_key_connect,
            logit_key_spawn=logit_key_spawn
        )
        
        # --- 2. POMO (Multi-Start) 준비 ---
        num_starts, start_nodes_idx = env.select_start_nodes(td)
        if num_starts == 0:
             # (B, 1) 형태의 0점 리워드 반환
            zero_reward = torch.zeros(td.batch_size[0], 1, device=td.device)
            return {"reward": zero_reward} # (POMO 시작 불가)

        num_total_loads = env.generator.num_loads
        batch_size = td.batch_size[0]
        
        # (B) -> (B * num_starts)
        td_expanded_view = batchify(td, num_starts)
        td = td_expanded_view
        cache = cache.batchify(num_starts) # 캐시도 확장

        # POMO 시작: 첫 액션(Load 선택)을 환경에 강제 적용
        first_action_tensor = start_nodes_idx.repeat(batch_size).unsqueeze(-1)
        
        # (POMO의 첫 스텝은 env._reset에서 처리되도록 solver_env.py에서 구현 필요)
        # (여기서는 td가 이미 첫 Load가 Head로 설정된 상태라고 가정합니다.)
        
        # --- 3. 디코딩 루프 ---
        log_probs: List[torch.Tensor] = []
        actions: List[Dict[str, torch.Tensor]] = []
        rewards: List[torch.Tensor] = []
        first_value: torch.Tensor = None
        
        decoding_step = 0
        while not td["done"].all():
            decoding_step += 1
            if pbar and log_mode == 'progress':
                # (진행률 표시: 0번 샘플 기준)
                unconnected = td['unconnected_loads_mask'][0].sum().item()
                connected = num_total_loads - unconnected
                pbar.set_description(f"{base_desc} | {status_msg} | Loads {connected}/{num_total_loads}")

            # 1. 디코더 호출 (4개 텐서 반환)
            logits_type, logits_connect, logits_spawn, value = self.decoder(td, cache)
            
            # A2C를 위해 첫 스텝의 Value(가치) 저장
            if decoding_step == 1:
                first_value = value.squeeze(-1) # (B * N_loads, 1) -> (B * N_loads)
            
            # 2. 환경에서 3종 마스크 가져오기
            # (solver_env.py가 반환할 마스크 딕셔너리)
            with torch.no_grad():
                masks: Dict[str, torch.Tensor] = env.get_action_mask(td)
            
            # 3. 3개 헤드에서 각각 샘플링
            action_type, log_prob_type = self._sample_action(
                logits_type, masks["mask_type"], decode_type
            )
            action_connect, log_prob_connect = self._sample_action(
                logits_connect, masks["mask_connect"], decode_type
            )
            action_spawn, log_prob_spawn = self._sample_action(
                logits_spawn, masks["mask_spawn"], decode_type
            )

            # 4. Parameterized Action Log Prob 결합
            final_log_prob = self._combine_log_probs(
                log_prob_type, action_type, 
                log_prob_connect, log_prob_spawn
            )
            
            # 5. 환경에 전달할 액션 딕셔너리 생성
            action_dict = {
                "action_type": action_type.unsqueeze(-1),
                "connect_target": action_connect.unsqueeze(-1),
                "spawn_template": action_spawn.unsqueeze(-1),
            }
            
            # [START]: 'detail' 모드 액션 로깅 (수정됨)
            if log_fn and log_mode == 'detail':
                # (첫 번째 샘플(B=0) 기준으로 로그 출력)
                sample_idx = 0
                if sample_idx < td.batch_size[0]:
                    current_head = td["trajectory_head"][sample_idx].item()
                    
                    # --- 1. 확률 분포 계산 ---
                    # (위에서 정의한 _get_masked_probs 사용)
                    probs_type = self._get_masked_probs(logits_type[sample_idx], masks["mask_type"][sample_idx])
                    probs_connect = self._get_masked_probs(logits_connect[sample_idx], masks["mask_connect"][sample_idx])
                    probs_spawn = self._get_masked_probs(logits_spawn[sample_idx], masks["mask_spawn"][sample_idx])

                    # --- 2. 이름 매핑 준비 ---
                    # (환경 설정에서 정적 이름 목록 가져오기)
                    node_names = env.generator.config.node_names
                    def get_name(idx):
                        if 0 <= idx < len(node_names): return node_names[idx]
                        return f"Spawned_Node_{idx}" # 동적 생성된 노드는 인덱스로 표시

                    head_name = get_name(current_head)
                    
                    log_fn(f"\n[Step {decoding_step:02d}] Current Head: {head_name} (idx: {current_head})")

                    # --- 3. Action Type 확률 출력 ---
                    p_conn = probs_type[0].item()
                    p_spwn = probs_type[1].item()
                    
                    chosen_type = action_type[sample_idx].item()
                    type_str = "Connect" if chosen_type == 0 else "Spawn"
                    
                    log_fn(f"  📊 Action Type Probabilities:")
                    log_fn(f"     - Connect: {p_conn*100:.2f}% {'👈 Selected' if chosen_type==0 else ''}")
                    log_fn(f"     - Spawn  : {p_spwn*100:.2f}% {'👈 Selected' if chosen_type==1 else ''}")

                    # --- 4. 상세 후보 확률 출력 ---
                    
                    # (A) Connect 후보들
                    if masks["mask_type"][sample_idx, 0]: # Connect가 가능한 경우만
                        log_fn(f"  🔗 Connect Candidates (P(Target | Connect)):")
                        valid_connect_indices = torch.where(masks["mask_connect"][sample_idx])[0]
                        
                        # 확률순 정렬
                        cand_probs = []
                        for idx in valid_connect_indices:
                            prob = probs_connect[idx].item()
                            cand_probs.append((prob, idx.item()))
                        cand_probs.sort(key=lambda x: x[0], reverse=True)

                        for prob, idx in cand_probs:
                            name = get_name(idx)
                            is_picked = (chosen_type == 0 and action_connect[sample_idx].item() == idx)
                            log_fn(f"     - {name:<25} : {prob*100:.2f}% {'✅' if is_picked else ''}")
                    
                    # (B) Spawn 후보들
                    if masks["mask_type"][sample_idx, 1]: # Spawn이 가능한 경우만
                        log_fn(f"  📦 Spawn Candidates (P(Template | Spawn)):")
                        valid_spawn_indices = torch.where(masks["mask_spawn"][sample_idx])[0]
                        
                        cand_probs = []
                        for idx in valid_spawn_indices:
                            prob = probs_spawn[idx].item()
                            cand_probs.append((prob, idx.item()))
                        cand_probs.sort(key=lambda x: x[0], reverse=True)

                        for prob, idx in cand_probs:
                            name = get_name(idx)
                            is_picked = (chosen_type == 1 and action_spawn[sample_idx].item() == idx)
                            log_fn(f"     - {name:<25} : {prob*100:.2f}% {'✅' if is_picked else ''}")

                    log_fn("-" * 60)
            # [END]: 'detail' 모드 액션 로깅

            # 6. 환경 스텝 실행
            with torch.no_grad():
                td.set("action", action_dict)
                output_td = env.step(td)
            
            reward = output_td["reward"]
            td = output_td["next"]
            
            # 7. A2C 학습을 위한 데이터 수집
            log_probs.append(final_log_prob)
            actions.append(action_dict)
            rewards.append(reward)

        # 8. 최종 결과 취합
        if not rewards:
            # (디코딩 루프가 1번도 돌지 않은 경우 - 예: 이미 완료된 상태)
            B_total = td.batch_size[0]
            dummy_reward = torch.zeros(B_total, 1, device=td.device)
            dummy_log_prob = torch.zeros(B_total, device=td.device)
            dummy_value = torch.zeros(B_total, 1, device=td.device)
            return {
                "reward": dummy_reward,
                "log_likelihood": dummy_log_prob,
                "actions": [],
                "value": dummy_value,
            }

        # (B_total, T) -> (B_total, 1)
        total_reward = torch.stack(rewards, 1).sum(1)
        # (B_total, T) -> (B_total)
        total_log_likelihood = torch.stack(log_probs, 1).sum(1)

        # [추가] 최종 상태에서 비용 정보 추출
        final_bom_cost = td["current_cost"].squeeze(-1)
        final_sleep_cost = td["sleep_cost"].squeeze(-1)


        result = {
            "reward": total_reward,
            "log_likelihood": total_log_likelihood,
            "actions": actions,  # (디버깅용)
            "value": first_value,
            "bom_cost": final_bom_cost, # [추가]
            "sleep_cost": final_sleep_cost, # [추가]
        }

        if return_final_td:
            # 시각화/디버깅용 최종 상태는 GPU 전체 TensorDict를 통째로
            # clone() 하는 대신,
            #  - 그래디언트 연결을 끊고(detach)
            #  - 필요한 키만 골라서
            #  - CPU 메모리로만 저장한다.
            #
            # visualize_result()에서 사용하는 키:
            #   - "nodes"
            #   - "adj_matrix"
            #   - "is_active_mask"
            final_td_cpu = TensorDict(
                {
                    "nodes": td["nodes"].detach().cpu(),
                    "adj_matrix": td["adj_matrix"].detach().cpu(),
                    "is_active_mask": td["is_active_mask"].detach().cpu(),
                },
                batch_size=td.batch_size,
            )
            result["final_td"] = final_td_cpu


        return result