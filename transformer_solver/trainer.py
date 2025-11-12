# transformer_solver/trainer.py

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader

from tqdm import tqdm
import os
import time
from datetime import datetime
import logging
from collections import defaultdict
import json

# --- 핵심 모듈 임포트 ---
from .model import PocatModel, PrecomputedCache, reshape_by_heads
from .solver_env import PocatEnv, BATTERY_NODE_IDX
from .expert_dataset import ExpertReplayDataset, expert_collate_fn
from .utils.common import TimeEstimator, clip_grad_norms, unbatchify, batchify

# --- 시각화 모듈 임포트 ---
from graphviz import Digraph
from common.data_classes import LDO, BuckConverter # (common)
from .definitions import FEATURE_INDEX, NODE_TYPE_LOAD, NODE_TYPE_IC


def update_progress(pbar, metrics, step):
    """ tqdm 진행률 표시줄을 업데이트합니다. """
    if pbar is None:
        return
    
    metrics_str = (
        f"Loss: {metrics['Loss']:.4f} "
        f"($Avg: {metrics['Avg Cost']:.2f}, $Min: {metrics['Min Cost']:.2f})"
    )
    pbar.set_postfix_str(metrics_str, refresh=False)
    pbar.update(1)


def cal_model_size(model, log_func):
    """ 모델의 파라미터 및 버퍼 크기를 계산하여 로그에 기록합니다. """
    param_count = sum(p.nelement() for p in model.parameters() if p.requires_grad)
    buffer_count = sum(b.nelement() for b in model.buffers())
    log_func(f'모델 파라미터 수: {param_count:,}')
    log_func(f'모델 버퍼 수: {buffer_count:,}')

class PocatTrainer:
    """
    PocatModel과 PocatEnv를 사용하여 훈련, 검증, 테스트를
    수행하는 메인 트레이너 클래스입니다. (A2C 기반)
    """
    
    def __init__(self, args, env: PocatEnv, device: str):
        self.args = args
        self.env = env
        self.is_ddp = args.ddp
        self.local_rank = args.local_rank
        self.device = device

        self.result_dir = args.result_dir
        self.log = args.log

        # --- 1. 모델 초기화 및 DDP 래핑 ---
        self.model = PocatModel(**args.model_params).to(self.device)
        
        if self.is_ddp:
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank], 
                find_unused_parameters=False # (모델은 모든 파라미터 사용)
            )
        
        if self.local_rank <= 0:
            cal_model_size(self.model, self.log)
        
        # --- 2. 옵티마이저 및 스케줄러 ---
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(args.optimizer_params['optimizer']['lr']),
            weight_decay=float(args.optimizer_params['optimizer'].get('weight_decay', 0)),
        )
        
        if args.optimizer_params['scheduler']['name'] == 'MultiStepLR':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=args.optimizer_params['scheduler']['milestones'],
                gamma=args.optimizer_params['scheduler']['gamma']
            )
        else:
            raise NotImplementedError
            
        self.start_epoch = 1

        # --- 3. 모델 로드 (Checkpoint) ---
        if args.load_path is not None:
            self.log(f"모델 체크포인트 로드 중: {args.load_path}")
            try:
                checkpoint = torch.load(args.load_path, map_location=device)
                
                # DDP/일반 모델 상태 호환 로드
                model_to_load = self.model.module if self.is_ddp else self.model
                model_to_load.load_state_dict(checkpoint['model_state_dict'])
                
                if not args.test_only: # 훈련 재개 시
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.start_epoch = checkpoint['epoch'] + 1
                self.log("모델 로드 완료.")
            except Exception as e:
                self.log(f"❌ 모델 로드 실패: {e}. 무작위 초기화로 시작합니다.")

        self.time_estimator = TimeEstimator(log_fn=self.log)

        # --- 4. 검증(Evaluate)용 데이터셋 ---
        self.eval_batch_size = getattr(args, "eval_batch_size", 128)
        if self.local_rank <= 0: # 0번 GPU에서만 생성
            with torch.no_grad():
                self._eval_td_fixed = self.env.generator(
                    batch_size=self.eval_batch_size
                ).to(self.device)
        self.best_eval_bom = float("inf")

    def pretrain_critic(self, expert_data_path: str, pretrain_epochs: int = 5):
        """
        '정답지(Expert)' 데이터셋을 사용하여 A2C 모델의 Critic(Value Head)만
        지도학습 방식으로 사전훈련합니다.
        """
        args = self.args
        self.log("=================================================================")
        self.log(f"🧠 Critic 사전훈련(Pre-training) 시작...")
        
        try:
            expert_dataset = ExpertReplayDataset(
                expert_data_path=expert_data_path, 
                env=self.env, 
                device=self.device
            )
            if len(expert_dataset) == 0:
                self.log("❌ 오류: '정답지' 데이터셋이 비어있어 사전훈련을 건너뜁니다.")
                return
            
            expert_loader = DataLoader(
                expert_dataset,
                batch_size=args.batch_size, # 훈련 배치 크기 재사용
                shuffle=True,
                num_workers=0,
                collate_fn=expert_collate_fn # TensorDict용 커스텀 Collate
            )
        except Exception as e:
            self.log(f"❌ 오류: '정답지' 데이터셋 로드 실패: {e}")
            return

        # Critic 파라미터만 학습하는 별도의 옵티마이저 생성
        model_to_train = self.model.module if self.is_ddp else self.model
        critic_params = list(model_to_train.decoder.value_head.parameters()) + \
                        list(model_to_train.decoder.Wq_context.parameters()) + \
                        list(model_to_train.decoder.multi_head_combine.parameters())
                        
        critic_optimizer = torch.optim.AdamW(
            critic_params,
            lr=float(args.optimizer_params['optimizer']['lr'])
        )

        self.model.train()

        for epoch in range(1, pretrain_epochs + 1):
            pbar = tqdm(expert_loader, desc=f"Critic Pre-train Epoch {epoch}/{pretrain_epochs}", dynamic_ncols=True)
            total_v_loss = 0
            
            for state_td_batch, target_reward_batch in pbar:
                critic_optimizer.zero_grad()
                
                # (B, 1, ...) -> (B, ...)
                state_td_batch = state_td_batch.squeeze(1)
                
                # --- 모델 인코딩 및 캐시 생성 ---
                prompt_embedding = model_to_train.prompt_net(
                    state_td_batch["scalar_prompt_features"], 
                    state_td_batch["matrix_prompt_features"]
                )
                encoded_nodes = model_to_train.encoder(state_td_batch, prompt_embedding)
                
                glimpse_key = reshape_by_heads(model_to_train.decoder.Wk_glimpse(encoded_nodes), model_to_train.decoder.head_num)
                glimpse_val = reshape_by_heads(model_to_train.decoder.Wv_glimpse(encoded_nodes), model_to_train.decoder.head_num)
                logit_key_connect = model_to_train.decoder.Wk_connect_logit(encoded_nodes).transpose(1, 2)
                logit_key_spawn = model_to_train.decoder.Wk_spawn_logit(encoded_nodes).transpose(1, 2)
                
                cache = PrecomputedCache(
                    encoded_nodes, glimpse_key, glimpse_val, 
                    logit_key_connect, logit_key_spawn
                )
                
                # --- 디코더 호출 (Value만 사용) ---
                _, _, _, predicted_value = model_to_train.decoder(state_td_batch, cache)
                
                # V_Loss 계산: Critic의 예측 vs "정답지"의 실제 보상
                critic_loss = F.mse_loss(predicted_value, target_reward_batch)
                
                critic_loss.backward()
                critic_optimizer.step()
                
                total_v_loss += critic_loss.item()
                pbar.set_postfix({"V_Loss (Pre)": f"{critic_loss.item():.4f}"})

            self.log(f"Critic Pre-train Epoch {epoch} | Avg V_Loss: {total_v_loss / len(expert_loader):.4f}")

        self.log("✅ Critic 사전훈련 완료.")
        self.log("=================================================================")

    def run(self):
        """ 메인 훈련 루프 (A2C) """
        args = self.args
        self.time_estimator.reset(self.start_epoch)
        
        if args.test_only:
            self.test()
            return

        for epoch in range(self.start_epoch, args.trainer_params['epochs'] + 1):
            if self.local_rank <= 0:
                self.log('=' * 60)
            
            self.model.train()
            
            # (DDP) DDP Sampler가 에폭마다 시드를 변경하도록 설정
            if self.is_ddp and hasattr(self.env_dataset, 'sampler'):
                self.env_dataset.sampler.set_epoch(epoch)
            
            total_steps = args.trainer_params['train_step']
            
            # (DDP) 0번 GPU에서만 tqdm 진행률 표시
            train_pbar = None
            if self.local_rank <= 0:
                train_pbar = tqdm(
                    total=total_steps,
                    desc=f"Epoch {epoch}",
                    dynamic_ncols=True,
                )
            
            total_loss = 0.0
            total_cost = 0.0
            total_policy_loss = 0.0
            total_critic_loss = 0.0
            min_epoch_cost = float('inf')

            for step in range(1, total_steps + 1):
                self.optimizer.zero_grad()
                
                # 1. 환경 리셋
                # (DDP 사용 시, 각 GPU는 B/N 개의 배치를 처리)
                td = self.env.reset(batch_size=args.batch_size)
                
                # 2. POMO (Multi-Start) 확장
                num_starts = self.env.generator.num_loads
                if args.num_pomo_samples > 1:
                    # (Load 개수(num_starts)만큼만 확장)
                    td = batchify(td, num_starts)
                
                # 3. 모델 포워드 (솔루션 생성)
                out = self.model(
                    td, self.env, decode_type='sampling', pbar=train_pbar,
                    status_msg=f"Epoch {epoch}", log_fn=self.log,
                    log_idx=args.log_idx, log_mode=args.log_mode
                )
                
                # 4. A2C 손실 계산
                # (B, N_pomo)
                reward = out["reward"].view(-1, num_starts)
                log_likelihood = out["log_likelihood"].view(-1, num_starts)
                value = out["value"].view(-1, num_starts)

                # Critic Loss (V(s)가 실제 보상(G)을 예측하도록)
                critic_loss = F.mse_loss(value, reward)

                # Policy Loss (Actor)
                advantage = reward - value.detach() # Baseline = V(s)
                policy_loss = -(advantage * log_likelihood).mean()

                # Total Loss (A2C)
                loss = policy_loss + 0.5 * critic_loss
                
                # 5. 역전파 및 가중치 업데이트
                loss.backward()
                
                max_norm = float(self.args.optimizer_params.get('max_grad_norm', 0))
                if max_norm > 0:
                    clip_grad_norms(self.optimizer.param_groups, max_norm=max_norm)
                self.optimizer.step()

                # (DDP) 모든 GPU의 통계를 집계
                if self.is_ddp:
                    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                    dist.all_reduce(policy_loss, op=dist.ReduceOp.AVG)
                    dist.all_reduce(critic_loss, op=dist.ReduceOp.AVG)
                    # (min_cost는 all_reduce(op=dist.ReduceOp.MIN) 필요)
                
                # (DDP) 0번 GPU에서만 로그 기록
                if self.local_rank <= 0:
                    avg_cost = -reward.mean().item()
                    min_batch_cost = -reward.max().item()
                    min_epoch_cost = min(min_epoch_cost, min_batch_cost)

                    total_loss += loss.item()
                    total_cost += avg_cost
                    total_policy_loss += policy_loss.item()
                    total_critic_loss += critic_loss.item()

                    update_progress(
                        train_pbar,
                        {
                            "Loss": loss.item(),
                            "Avg Cost": total_cost / step,
                            "Min Cost": min_epoch_cost,
                        },
                        step
                    )

            if train_pbar:
                train_pbar.close()

            # (DDP) 0번 GPU에서만 에폭 요약, 검증, 저장
            if self.local_rank <= 0:
                epoch_summary = (
                    f"Epoch {epoch}/{args.trainer_params['epochs']} | "
                    f"Avg Loss {total_loss / total_steps:.4f} | "
                    f"P_Loss {total_policy_loss / total_steps:.4f} | "
                    f"V_Loss {total_critic_loss / total_steps:.4f} | "
                    f"Min Cost ${min_epoch_cost:.2f}"
                )
                tqdm.write(epoch_summary)
                self.log(epoch_summary)
                
                # --- 검증 (Evaluate) ---
                val = self.evaluate(epoch)
                self.log(f"[Eval] Epoch {epoch} | Avg BOM ${val['avg_bom']:.2f} | Min BOM ${val['min_bom']:.2f}")

                # --- 체크포인트 저장 ---
                if (epoch % args.trainer_params['model_save_interval'] == 0) or \
                   (epoch == args.trainer_params['epochs']):
                       
                    save_path = os.path.join(args.result_dir, f'epoch-{epoch}.pth')
                    self.log(f"모델 저장 중... (Epoch {epoch} -> {save_path})")
                    self._run_test_visualization(epoch, is_best=False) # 시각화
                    
                    model_state_dict = self.model.module.state_dict() if self.is_ddp else self.model.state_dict()
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_state_dict,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, save_path)

            self.scheduler.step()

            if self.local_rank <= 0:
                self.time_estimator.print_est_time(epoch, args.trainer_params['epochs'])
            
            if self.is_ddp:
                dist.barrier() # 에폭 종료 시 모든 GPU 동기화

        if self.local_rank <= 0:
            self.log(" *** 훈련 완료 *** ")


    @torch.no_grad()
    def evaluate(self, epoch: int):
        """ 고정된 검증 셋(Validation Set)에 대해 Greedy 탐색을 수행합니다. """
        self.model.eval()
        
        # (고정된 검증 데이터셋 사용)
        td_eval = self.env._reset(self._eval_td_fixed.clone())
        
        # POMO (Load 개수만큼) 확장
        eval_samples, start_nodes_idx = self.env.select_start_nodes(td_eval)
        if eval_samples > 1:
            td_eval = batchify(td_eval, eval_samples)
            # (pom_start 로직은 env._reset에서 처리되었다고 가정)

        out = self.model(
            td_eval, self.env, decode_type='greedy',
            pbar=None, status_msg="Eval",
            log_fn=self.log, log_idx=self.args.log_idx, log_mode='progress'
        )

        # (B, N_pomo)
        reward = out["reward"].view(self.eval_batch_size, eval_samples)
        # 인스턴스별 최고 점수 (B,)
        best_reward_per_instance = reward.max(dim=1)[0]

        avg_bom = -best_reward_per_instance.mean().item()
        min_bom = -best_reward_per_instance.max().item()

        # CSV 로그
        csv_path = os.path.join(self.result_dir, "val_log.csv")
        header = not os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8") as f:
            if header: f.write("epoch,avg_bom,min_bom,decode_type\n")
            f.write(f"{epoch},{avg_bom:.4f},{min_bom:.4f},greedy\n")

        # 최고 성능 모델 저장
        if avg_bom < self.best_eval_bom:
            self.best_eval_bom = avg_bom
            save_path = os.path.join(self.result_dir, "best_cost.pth")
            self.log(f"[Eval] ✅ 새 최고 성능 달성 ${avg_bom:.2f} (min=${min_bom:.2f}) -> {save_path} 저장")
            
            # 테스트 시각화 실행
            self._run_test_visualization(epoch, is_best=True)
            
            model_state_dict = self.model.module.state_dict() if self.is_ddp else self.model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, save_path)

        return {"avg_bom": avg_bom, "min_bom": min_bom}

    def test(self):
        """ 테스트 모드 (추론)를 실행합니다. """
        self.model.eval()
        self.log("=" * 60)
        self.log("🔬 테스트 모드 (추론) 시작...")
        self._run_test_visualization(epoch=0, is_best=False)
        self.log("=" * 60)

    @torch.no_grad()
    def _run_test_visualization(self, epoch: int, is_best: bool = False):
        """
        단일 인스턴스에 대해 추론을 실행하고,
        최종 텐서(TensorDict) 상태를 기반으로 파워트리 시각화(PNG)를 저장합니다.
        """
        self.model.eval()
        args = self.args

        if is_best:
            log_prefix = f"[Test Viz @ Epoch {epoch} (BEST)]"
            filename_prefix = f"epoch_{epoch}_best"
        elif epoch > 0:
            log_prefix = f"[Test Viz @ Epoch {epoch}]"
            filename_prefix = f"epoch_{epoch}"
        else:
            log_prefix = "[Test Viz (Standalone)]"
            filename_prefix = "test_solution"

        self.log(f"{log_prefix} 추론 및 시각화 시작...")

        # 1. 단일 배치(B=1)로 환경 리셋
        td = self.env.reset(batch_size=1)
        
        # 2. POMO 확장
        test_samples, start_nodes_idx = self.env.select_start_nodes(td)
        if args.test_num_pomo_samples > test_samples:
             self.log(f"Warning: test_num_pomo_samples({args.test_num_pomo_samples})가 Load 개수({test_samples})보다 큽니다.")
        
        # (Load 개수만큼만 확장)
        td = batchify(td, test_samples)
        
        # (POMO 시작 상태 적용 - solver_env가 _reset에서 처리)
        
        pbar_desc = f"Solving (Mode: {args.decode_type}, Samples: {test_samples})"
        pbar = tqdm(total=1, desc=pbar_desc, dynamic_ncols=True)
        
        # 3. 모델 추론
        out = self.model(
            td, self.env, decode_type=args.decode_type, pbar=pbar, 
            log_fn=self.log, log_idx=args.log_idx,
            log_mode=args.log_mode
        )
        pbar.close()

        # 4. 최고 성능 솔루션 선택
        reward = out['reward'] # (B_total,)
        
        best_idx = reward.argmax()
        final_cost = -reward[best_idx].item()
        
        # 5. 최종 상태(TensorDict) 추출
        # (td는 env.step()에 의해 in-place로 수정되었으므로,
        #  model()이 반환된 후의 td가 최종 상태입니다)
        final_td_instance = td[best_idx].clone()

        # 6. POMO 시작 노드 이름 찾기
        best_start_node_local_idx = best_idx % test_samples
        best_start_node_idx = start_nodes_idx[best_start_node_local_idx].item()
        best_start_node_name = self.env.generator.config.node_names[best_start_node_idx]
        
        self.log(f"추론 완료 (Cost: ${final_cost:.4f}, Start: '{best_start_node_name}')")

        # 7. 시각화 실행
        self.visualize_result(
            final_td_instance, 
            final_cost, 
            best_start_node_name, 
            filename_prefix
        )
        self.log(f"{log_prefix} 시각화 다이어그램 저장 완료.")

    def visualize_result(self, 
                         final_td: TensorDict, 
                         final_cost: float, 
                         best_start_node_name: str, 
                         filename_prefix: str = "solution"):
        """
        Lazy Spawn에 맞게 수정된 시각화 함수.
        
        최종 TensorDict 상태(final_td)의 'is_active_mask'와 'adj_matrix'를
        읽어 활성화된 노드와 엣지만 그립니다.
        """
        if self.result_dir is None: return
        os.makedirs(self.result_dir, exist_ok=True)

        # 1. 정보 추출
        node_names = self.env.generator.config.node_names # (N_max,)
        all_nodes_features = final_td["nodes"] # (N_max, D)
        adj_matrix = final_td["adj_matrix"] # (N_max, N_max)
        is_active = final_td["is_active_mask"] # (N_max,)
        
        battery_conf = self.env.generator.config.battery
        constraints = self.env.generator.config.constraints

        # 2. Graphviz 객체 생성
        dot = Digraph(comment=f"Power Tree - Cost ${final_cost:.4f}")
        dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
        dot.attr(rankdir='LR', label=f"Transformer Solution (Start: {best_start_node_name})\nCost: ${final_cost:.4f}", labelloc='t')

        # 3. 활성화된(Active) 노드만 순회
        active_indices = torch.where(is_active)[0]
        
        # (암전류/독립레일 계산을 위한 사전 작업)
        active_adj_matrix = adj_matrix[is_active][:, is_active]
        active_nodes_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(active_indices)}
        
        # 4. 노드 추가 (Dot)
        for node_idx_tensor in active_indices:
            node_idx = node_idx_tensor.item()
            node_feat = all_nodes_features[node_idx]
            node_type = self.env.node_type_tensor[node_idx]
            node_name = node_names[node_idx] if node_idx < len(node_names) else f"Spawned_IC_{node_idx}"
            
            label = ""
            
            if node_type == NODE_TYPE_BATTERY:
                # (암전류 계산은 복잡하므로 여기서는 생략, 총 전력만 표기)
                label = f"🔋 {node_name}\n\nCost: ${final_cost:.2f}"
                dot.node(node_name, label, shape='Mdiamond', color='darkgreen', fillcolor='white')
            
            elif node_type == NODE_TYPE_LOAD:
                current_active_ma = node_feat[FEATURE_INDEX["current_active"]].item() * 1000
                current_sleep_ua = node_feat[FEATURE_INDEX["current_sleep"]].item() * 1000000
                label = f"💡 {node_name}\nActive: {current_active_ma:.1f}mA"
                if current_sleep_ua > 0:
                    label += f"\nSleep: {current_sleep_ua:,.1f}µA"
                dot.node(node_name, label, color='dimgray', fillcolor='white')

            elif node_type == NODE_TYPE_IC:
                # (스폰된 IC)
                i_out_ma = node_feat[FEATURE_INDEX["current_out"]].item() * 1000
                tj = node_feat[FEATURE_INDEX["junction_temp"]].item()
                tj_max = node_feat[FEATURE_INDEX["t_junction_max"]].item()
                cost = node_feat[FEATURE_INDEX["cost"]].item()
                
                thermal_margin = tj_max - tj
                node_color = 'blue'
                if thermal_margin < 10: node_color = 'red'
                elif thermal_margin < 25: node_color = 'orange'
                
                label = (f"📦 {node_name.split('@')[0]}\n\n"
                         f"Iout: {i_out_ma:.1f}mA (Active)\n"
                         f"Tj: {tj:.1f}°C (Max: {tj_max}°C)\n"
                         f"Cost: ${cost:.2f}")
                dot.node(node_name, label, color=node_color, fillcolor='lightgray', style='rounded,filled,dashed', penwidth='3')

        # 5. 엣지 추가 (Dot)
        parent_indices, child_indices = adj_matrix.nonzero(as_tuple=True)
        for p_idx, c_idx in zip(parent_indices, child_indices):
            # 두 노드 모두 활성화된 경우에만 엣지를 그림
            if is_active[p_idx] and is_active[c_idx]:
                p_name = node_names[p_idx] if p_idx < len(node_names) else f"Spawned_IC_{p_idx}"
                c_name = node_names[c_idx] if c_idx < len(node_names) else f"Spawned_IC_{c_idx}"
                dot.edge(p_name, c_name)
        
        # 6. 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_cost_{final_cost:.4f}_{timestamp}"
        output_path = os.path.join(self.result_dir, filename)
        
        try:
            dot.render(output_path, view=False, format='png', cleanup=True)
            self.log(f"✅ 상세 시각화 다이어그램을 {output_path}.png 파일로 저장했습니다.")
        except Exception as e:
            self.log(f"❌ 시각화 렌더링 실패. (Graphviz 설치 확인 필요): {e}")