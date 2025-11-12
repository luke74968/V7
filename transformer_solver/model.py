# transformer_solver/model.py

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
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key_connect: torch.Tensor # 'Connect' 포인터용 Key
    logit_key_spawn: torch.Tensor   # 'Spawn' 포인터용 Key

    def batchify(self, num_starts: int):
        """ POMO 샘플링을 위해 캐시를 N_starts 배수만큼 복제합니다. """
        return PrecomputedCache(
            batchify(self.node_embeddings, num_starts),
            batchify(self.glimpse_key, num_starts),
            batchify(self.glimpse_val, num_starts),
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
        
        node_embeddings += self.embedding_is_active(active_ids)
        node_embeddings += self.embedding_is_template(template_ids)
        node_embeddings += self.embedding_can_spawn_into(spawn_ids)
        node_embeddings += self.embedding_rail_type(rail_ids)
        
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
    """
    Parameterized Action Decoder (4-Head).
    
    1. 현재 상태(Head, Global)로 컨텍스트 쿼리(q_vec) 생성
    2. q_vec을 4개의 헤드(Value, Type, Connect, Spawn)로 분배
    3. 4개의 동시 결정(Logits/Value)을 출력
    """
    def __init__(self, embedding_dim, head_num, qkv_dim, N_MAX, **model_params):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        self.N_MAX = N_MAX
        
        # 1. 컨텍스트 쿼리(q_vec) 생성용 MHA
        self.Wk_glimpse = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_glimpse = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_context = nn.Linear(embedding_dim + 3, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        
        # 2. 포인터 네트워크용 Key 생성 (인코더 임베딩을 변환)
        self.Wk_connect_logit = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk_spawn_logit = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # --- 3. 4-Heads (q_vec을 입력으로 받음) ---
        
        # 3a. Critic Head (A2C)
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1) # (B, 1, 1) -> (B, 1)
        )
        
        # 3b. Action Type Head (0: Connect, 1: Spawn)
        self.type_head = nn.Linear(embedding_dim, 2)
        
        # 3c. Connect Target Head (Pointer)
        self.connect_head = nn.Linear(embedding_dim, embedding_dim)
        
        # 3d. Spawn Template Head (Pointer)
        self.spawn_head = nn.Linear(embedding_dim, embedding_dim)


    def forward(self, td: TensorDict, cache: PrecomputedCache) -> Tuple[torch.Tensor, ...]:
        
        # 1. 디코더 쿼리(q_vec) 생성 (A2C Critic과 Actor가 공유)
        
        # (avg_current, unconnected_ratio, step_ratio) 3개 피처 생성
        avg_current = td["nodes"][..., FEATURE_INDEX["current_out"]].clone().mean(dim=1, keepdim=True)
        unconnected_ratio = td["unconnected_loads_mask"].clone().float().mean(dim=1, keepdim=True)
        step_ratio = td["step_count"].clone().float() / (2 * self.N_MAX)

        state_features = torch.cat([avg_current, unconnected_ratio, step_ratio], dim=1)

        # 현재 Head 노드의 임베딩 가져오기
        # NOTE:
        #   td["trajectory_head"]는 환경 단계(env.step)에서 in-place로 갱신된다.
        #   squeeze(-1)만 호출하면 동일한 저장소(storage)를 공유하게 되며,
        #   이후 env.step에서 해당 텐서를 수정할 때 autograd가 저장해둔 인덱스가
        #   변경되어 버려 역전파 시 "modified by an inplace operation" 오류가 발생한다.
        #   clone()으로 독립적인 텐서를 만들어 안전하게 사용한다.
        head_idx = td["trajectory_head"].detach().squeeze(-1).clone()
        batch_indices = torch.arange(td.batch_size[0], device=head_idx.device)
        head_emb = cache.node_embeddings[batch_indices, head_idx]
        
        # 쿼리 입력: (Head 임베딩 + 3개 상태 피처)
        query_input = torch.cat([head_emb, state_features], dim=1)
        
        # 컨텍스트 쿼리(q) 생성
        q_context = reshape_by_heads(self.Wq_context(query_input.unsqueeze(1)), self.head_num)
        
        # MHA 수행
        mha_out = multi_head_attention(q_context, cache.glimpse_key, cache.glimpse_val)
        
        # (B, 1, D) - 모든 헤드가 공유할 최종 컨텍스트 벡터
        q_vec = self.multi_head_combine(mha_out) 

        # --- 2. 4개의 헤드로 q_vec 분배 ---
        
        # 2a. Critic Value (B, 1)
        value = self.value_head(q_vec).squeeze(-1)
        
        # 2b. Action Type Logits (B, 2)
        logits_action_type = self.type_head(q_vec).squeeze(1)
        
        # 2c. Connect Target Logits (B, N_MAX)
        query_connect = self.connect_head(q_vec) # (B, 1, D)
        logits_connect_target = torch.matmul(
            query_connect, cache.logit_key_connect
        ).squeeze(1) / (self.embedding_dim ** 0.5)
        
        # 2d. Spawn Template Logits (B, N_MAX)
        query_spawn = self.spawn_head(q_vec) # (B, 1, D)
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
                log_fn=None, log_idx: int = 0, log_mode: str = 'progress'
                ) -> Dict[str, torch.Tensor]:
        
        base_desc = pbar.desc.split(' | ')[0] if pbar else ""
        if pbar: pbar.set_description(f"{base_desc} | {status_msg} | ▶ Encoding")
        
        # --- 1. 인코딩 및 캐시 생성 ---
        prompt_embedding = self.prompt_net(td["scalar_prompt_features"], td["matrix_prompt_features"])
        encoded_nodes = self.encoder(td, prompt_embedding) # (B, N_MAX, D)
        
        # 디코더가 사용할 Key/Value 사전 계산
        glimpse_key = reshape_by_heads(self.decoder.Wk_glimpse(encoded_nodes), self.decoder.head_num)
        glimpse_val = reshape_by_heads(self.decoder.Wv_glimpse(encoded_nodes), self.decoder.head_num)
        
        # 포인터 헤드별 Key 생성
        logit_key_connect = self.decoder.Wk_connect_logit(encoded_nodes).transpose(1, 2)
        logit_key_spawn = self.decoder.Wk_spawn_logit(encoded_nodes).transpose(1, 2)
        
        cache = PrecomputedCache(
            node_embeddings=encoded_nodes,
            glimpse_key=glimpse_key,
            glimpse_val=glimpse_val,
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
        td = td_expanded_view.clone()
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
            
            # 6. 환경 스텝 실행
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

        return {
            "reward": total_reward,
            "log_likelihood": total_log_likelihood,
            "actions": actions, # (디버깅용)
            "value": first_value 
        }