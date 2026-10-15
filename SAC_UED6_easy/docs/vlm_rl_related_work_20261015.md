# VLM/LLM 의사결정과 하위 행동 실행: 관련 연구 조사

조사일: **2026-10-15**. 대상 구상: 상위 VLM이 군중 상황을 해석하여 로봇별 유입 차단·대피 유도 등의 임무를 배정하고, 하위 정책이 이동과 신호 행동을 실행하는 구조.

## 1. 조사 범위와 해석

원 논문(arXiv·학회 원문), 저자 프로젝트 및 공식 코드 설명을 기준으로 **관련 연구 50편**을 분류했다. 별도로 비VLM 군중 대피 시뮬레이션 1편을 경계 사례로 기록했다. 전수 문헌조사나 군중 대피에서의 최초성 검증은 아니다. 비교적 먼 연구도 인터페이스·피드백·학습 방식의 대안으로 포함했다.

- **[Certain]** 상위 언어/시각언어 모델과 하위 실행 정책을 분리하는 구조에는 다수의 선례가 있다. 그 결합 자체를 새로운 기여라고 주장하기 어렵다.
- **[Likely]** 이 프로젝트에서는 동적 군중에 대한 역할 분담, 유입 차단과 대피 유도의 시간적 조정, 실행 가능성에 따른 재할당 및 미지 지도 일반화가 연구 질문이 될 수 있다. 이 조합의 최초성·효과는 아직 입증하지 않았다.
- 연도는 주로 최초 공개 연도이며, 원문에서 확인한 학회 연도를 함께 표기한 항목도 있다. arXiv만 연결한 항목은 학회 게재 여부를 보증하지 않는다. 2026년 논문은 조사일 이전 공개 자료다.
- 핵심 관련 논문은 방법 절까지 확인했으나 전체 논문을 동일 깊이로 정독하거나 재현하지는 않았다. 아래는 구조 중심의 조사이며 벤치마크가 다른 성공률을 직접 순위화하지 않는다.

### 서로 다른 ‘VLM + RL’을 구분하기

| 구분 | 모델의 역할 | 이 프로젝트와의 관계 |
| --- | --- | --- |
| VLM → RL 실행 | 상위가 임무·스킬·목표를 선택, RL 정책이 수행 | 가장 직접적인 구조 |
| VLM → IL/VLA 실행 | 상위가 지시/경로/latent를 출력, 시연으로 학습한 정책이 수행 | RL 대신 실행기를 바꾼 구조적 선례 |
| VLM/LLM + 기하 계획 | 목표·제약·후보 점수를 제공, 탐색/MPC/제어기가 실행 | RL 없는 비교 대상 |
| VLA 직접 제어 | 시각·언어에서 행동 토큰/연속 행동을 예측 | 반드시 명시적 임무 계층이 있는 것은 아님 |
| RL로 VLM 추론 학습 | 상위 추론·계획을 RL로 개선 | 하위 SAC 학습과 다른 축 |
| VLM/LLM이 보상 설계·평가 | 학습 시 보상 코드·선호·점수를 생성 | 실행 시 임무를 정하는 방식과 다름 |

IL은 imitation learning(모방학습), MPC는 model predictive control, VLA는 vision-language-action이다. 시각 모델의 장면 설명을 텍스트 LLM에 주는 시스템을 이미지 입력 VLM 하나가 직접 계획하는 시스템과 동일시하지 않는다. 또한 모듈에 `policy`라는 이름이 있다는 이유만으로 RL로 분류하지 않는다.

## 2. A — 상위 판단과 RL 실행에 직접 가까운 연구 (7편)

| ID | 연구·출처 | 상위 출력 → 하위 실행 | 적용 및 주의점 |
| --- | --- | --- | --- |
| A1 | [SayCan: Do As I Can, Not As I Say (2022)](https://arxiv.org/abs/2204.01691) | LLM의 스킬 관련도와 RL로 학습한 affordance/value를 결합하여 스킬 선택 | 임무 의미와 실행 가능성을 결합하는 핵심 선례. 상위는 텍스트 LLM이므로 직접 이미지 VLM 계획과 구분. |
| A2 | [NaVILA: Legged Robot Vision-Language-Action Model for Navigation (2024; RSS 2025)](https://arxiv.org/abs/2412.04453) | VLM에서 적응한 VLA가 ‘앞으로 75cm’ 같은 중간 명령 생성 → 시각/지형 기반 RL 보행 정책 | 상위 의미 판단과 빠른 하위 실행을 분리. 하위 PPO 및 실물 실험 확인. 명령 수준은 대피 역할보다 낮다. [공식 프로젝트](https://navila-bot.github.io/) |
| A3 | [VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation (2023; ICRA 2024)](https://arxiv.org/abs/2312.03275) | 사전학습 VLM의 이미지–텍스트 점수로 value map과 frontier 목표 선정 → RL PointNav | VLM이 자연어 미션을 생성하지 않아도 목표 선택에 사용할 수 있음. PointNav는 VER로 사전학습하며, zero-shot은 모든 모듈이 무학습이라는 뜻이 아님. [공식 코드](https://github.com/rai-opensource/vlfm) |
| A4 | [RoBridge: A Hierarchical Architecture Bridging Cognition and Execution for General Robotic Manipulation (2025)](https://arxiv.org/abs/2505.01709) | VLM이 행동 종류·대상·목적지 및 IOR 생성 → embodied agent 실행 | 자유 문장 대신 mask·depth·행동 종류·제약의 구조화된 중간 표현. RL expert 학습과 IL/집약을 함께 쓰므로 순수 RL 실행기만의 사례로 단순화하지 않음. |
| A5 | [Hierarchical DLO Routing with Reinforcement Learning and In-Context Vision-language Models (2025; 2026 개정)](https://arxiv.org/abs/2510.19268) | VLM의 케이블/로프 라우팅 계획 → RL 스킬 실행 및 실패 복구 | ‘상위 VLM 계획 + 하위 RL 스킬’의 직접 선례. 대상은 정해진 조작 과제이며 동적 군중 검증은 아님. |
| A6 | [Hierarchical Vision-Language Planning for Multi-Step Humanoid Manipulation (2025)](https://arxiv.org/abs/2506.22827) | VLM 스킬 선택·완료 감시 → IL 스킬 → RL 전신 추종 제어 | 3계층 구조. VLM 바로 아래 모든 행동이 RL로 학습된다는 해석은 틀림. 완료 감시와 실행 역할 분리가 참고점. |
| A7 | [CoINS: Counterfactual Interactive Navigation via Skill-Aware VLM (2026)](https://arxiv.org/abs/2601.03956) | VLM이 스킬·상호작용 대상을 선택 → RL 상호작용 스킬과 RL 전신 제어 | 스킬 한계·공간 제약을 상위 입력에 명시. 일반 이동은 A*+pure pursuit, 상호작용과 전신 제어는 PPO인 혼합 구조다. 장애물을 치워 경로를 만드는 과제로 사람 안내와는 다름. |

## 3. B — VLM 판단과 학습된 행동을 분리한 계층형 VLA (7편)

| ID | 연구·출처 | 인터페이스·학습 방식 | 적용 및 주의점 |
| --- | --- | --- | --- |
| B1 | [Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models (ICML 2025)](https://proceedings.mlr.press/v267/shi25d.html) | 상위가 시각·지시·사용자 피드백으로 하위 지시 생성 → 하위 VLA | 미션 분해 및 중간 피드백 반영에 매우 가까움. 하위가 SAC인 논문은 아님. |
| B2 | [HAMSTER: Hierarchical Action Models For Open-World Robot Manipulation (2025)](https://arxiv.org/abs/2502.05485) | VLM이 거친 2D 말단 경로 → 3D-aware 학습 제어 정책 | 텍스트 외 중간 표현의 대안. 2D 경로의 좌표계·깊이 오차를 하위가 해석해야 함. |
| B3 | [ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning (2025)](https://arxiv.org/abs/2507.16815) | RL로 개선한 multimodal reasoning → visual-plan latent → 하위 행동 모델 | RL이 상위 추론 개선에 쓰이는 사례. 단순 ‘VLM 호출 + SAC 실행’과 다름. |
| B4 | [VINE: Hierarchical Vision Language Action Model Using Success and Failure Demonstrations (2025)](https://arxiv.org/abs/2512.03913) | 성공·실패 시연으로 feasibility를 학습하고 scene graph에서 하위 목표 tree search → 기존 스킬 실행 | 실패 자료를 실행 가능성 판단에 활용. HRL formalism이지만 온라인 저수준 RL 학습을 수행한다는 뜻은 아님. |
| B5 | [GR00T N1: An Open Foundation Model for Generalist Humanoid Robots (2025)](https://arxiv.org/abs/2503.14734) | vision-language System 2 → diffusion-transformer System 1, end-to-end 공동학습 | 빠른 행동부와 의미 해석부 분리. latent 결합이므로 명시적 로봇별 임무 배정과 구분. |
| B6 | [CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models (2025)](https://arxiv.org/abs/2503.22020) | 중간 목표 이미지 생성 → 짧은 행동 시퀀스 | 그림으로 전달하는 목표의 대안. 논문 이름의 reasoning을 RL 학습과 혼동하지 않음. |
| B7 | [What Matters in Orchestrating Robot Policies: A Systematic Study of Hierarchical VLA Agents (2026)](https://arxiv.org/abs/2606.10267) | options-style 틀에서 상위·하위 모델, 종료 조건, 관측 표현, 기억 설계를 비교 | 모델 선택뿐 아니라 **언제 임무를 바꿀지**가 독립적인 설계 축임을 보여주는 조사 우선순위 논문. 군중 결과로 바로 일반화하지 않음. |

## 4. C — 여러 로봇의 역할·과제 할당과 협업 (7편)

| ID | 연구·출처 | 의사결정 구조 | 적용 및 주의점 |
| --- | --- | --- | --- |
| C1 | [SMART-LLM: Smart Multi-Agent Robot Task Planning using Large Language Models (2023)](https://arxiv.org/abs/2309.10062) | task decomposition → coalition formation → task allocation | 로봇별 능력과 선후 관계에 맞춘 미션 배정. 직접 VLM 입력이나 RL 실행의 근거로 쓰지는 않음. |
| C2 | [CoELA: Building Cooperative Embodied Agents Modularly with Large Language Models (2023)](https://arxiv.org/abs/2307.02485) | 분산된 지각·기억·계획·통신·실행 모듈 | 부분 관측과 통신 비용이 있는 협업. 중앙 VLM 하나와 비교할 분산 구조. |
| C3 | [RoCo: Dialectic Multi-Robot Collaboration with Large Language Models (2023/2024)](https://project-roco.github.io/) | 로봇별 LLM 대화로 공동 계획·waypoint 제안, 환경/충돌 피드백으로 수정 | 협상과 충돌 제약. OWL-ViT 장면 설명을 쓰는 구조이며 모든 판단이 end-to-end VLM이나 RL인 것은 아님. |
| C4 | [COHERENT: Collaboration of Heterogeneous Multi-Robot System with Large Language Models (2024)](https://arxiv.org/abs/2409.15146) | 중앙 assigner → 로봇 executor → feedback → adjustment | 중앙 임무 배정과 하위 실행기의 실패 보고·재할당 구조가 직접적으로 가까움. |
| C5 | [EMOS: Embodiment-aware Heterogeneous Multi-robot Operating System with LLM Agents (2024; ICLR 2025)](https://arxiv.org/abs/2410.22662) | URDF·운동학 도구로 Robot Resume 생성, 능력에 맞는 협업 | 몸체·도달 범위·이동 능력 기반 배정. 동일 형태 로봇에서는 위치·부하·실행 성능 등 상태 의존 능력으로 확장 가능하나 이는 우리 제안임. |
| C6 | [LLaMAR: Long-Horizon Planning for Multi-Agent Robots in Partially Observable Environments (2024)](https://arxiv.org/abs/2407.10031) | plan → act → correct → verify; MAP-THOR 및 Search & Rescue 평가 | 부분 관측에서 실행 실패·완료를 반영하는 장기 협업. 수색·구조 평가가 있다는 것과 군중 대피 로봇이 검증됐다는 것은 다름. |
| C7 | [LaMMA-P: Generalizable Multi-Agent Long-Horizon Task Allocation and Planning with LM-Driven PDDL Planner (2024)](https://arxiv.org/abs/2409.20560) | LM과 PDDL heuristic search를 결합한 할당·계획 | 자연어 모델이 계획 전체를 자유 생성하는 방식과 비교할 기호 계획 대안. |

## 5. D — 시각·언어 계획과 스킬/기하 제어 결합 (8편)

| ID | 연구·출처 | 의사결정과 실행 | 적용 및 주의점 |
| --- | --- | --- | --- |
| D1 | [ViLa: Look Before You Leap: Unveiling the Power of GPT-4V in Robotic Vision-Language Planning (2023)](https://arxiv.org/abs/2311.17842) | 현재 영상과 목표로 스킬 계획, 첫 단계 실행 후 재관측 | 직접 VLM 계획의 명료한 선례. 스킬은 이미 있다고 가정하고 습득 방법을 핵심 기여로 다루지 않음. |
| D2 | [PaLM-E: An Embodied Multimodal Language Model (2023)](https://arxiv.org/abs/2303.03378) | 이미지·연속 센서 등 embodied 입력을 받아 언어 기반 로봇 계획 | VLM에 상태를 넣는 방법과 embodied grounding의 기반. RT-2와 같은 직접 행동 토큰 제어와 구분. |
| D3 | [Inner Monologue: Embodied Reasoning through Planning with Language Models (2022)](https://arxiv.org/abs/2207.05608) | 성공 판정·장면 설명·사람 피드백을 언어로 받아 재계획 | 계획을 한 번 세우고 끝내지 않는 실행 피드백 루프. 상위는 LLM. |
| D4 | [Code as Policies: Language Model Programs for Embodied Control (2022)](https://arxiv.org/abs/2209.07753) | LLM이 perception/control API 호출과 피드백 루프를 포함한 프로그램 생성 | 행동 API 및 명시적 제약을 통한 실행. 기존 RL 정책을 API로 노출하는 대안에 참고 가능. |
| D5 | [VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models (2023)](https://arxiv.org/abs/2307.05973) | LLM·VLM으로 affordance/constraint 3D map → model-based 궤적 계획 | 영역·경로·회피 제약 전달에 참고. value map을 SAC의 Bellman Q 함수와 동일시하지 않음. |
| D6 | [VLMPC: Vision-Language Model Predictive Control for Robotic Manipulation (2024)](https://arxiv.org/abs/2407.09829) | VLM이 후보 행동·비용에 관여, 비디오 예측과 MPC로 시퀀스 선택 | 후보 임무의 결과를 예측해 고르는 구조의 참고. 군중의 장기 반응 예측기가 검증됐다는 뜻은 아님. |
| D7 | [BUMBLE: Unifying Reasoning and Acting with Vision-Language Models for Building-wide Mobile Manipulation (2024)](https://arxiv.org/abs/2410.06237) | RGBD·스킬 라이브러리·두 층의 기억으로 건물 규모 행동 연결 | 장시간 실행의 기억과 실패 경험 관리. RL 실행기 연결 자체가 핵심인 논문은 아님. |
| D8 | [VeriGraph: Scene Graphs for Execution Verifiable Robot Planning (2024; ICRA 2026)](https://arxiv.org/abs/2411.10446) | VLM scene graph → 계획 → 제약 검증·수정 | 불가능한 임무를 실행 전에 거르는 명시적 검증. graph 정확성·제약 완전성이 전제임. |

## 6. E — 내비게이션의 목표·탐색·경로 의사결정 (6편)

| ID | 연구·출처 | 핵심 구조 | 적용 및 주의점 |
| --- | --- | --- | --- |
| E1 | [LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action (2022)](https://arxiv.org/abs/2207.04429) | 언어 랜드마크 추출, 시각 grounding, 사전학습 navigation 실행 결합 | 상위 언어 의미를 장소와 연결. 하위 학습 정책을 자동으로 RL이라 분류하지 않음. |
| E2 | [Visual Language Maps for Robot Navigation / VLMaps (2022; ICRA 2023)](https://arxiv.org/abs/2210.05714) | visual-language feature를 공간 지도에 연결해 언어 목표 위치 추출 | ‘북쪽 입구’, ‘위험 경계 바깥’ 같은 목표를 지도 좌표와 연결하는 설계 참고. |
| E3 | [Language Frontier Guide: Navigation with Large Language Models: Semantic Guesswork as a Heuristic for Planning (2023)](https://arxiv.org/abs/2310.10103) | LLM은 후보 frontier의 heuristic 점수, 기존 planner는 실행 | 상위 모델을 최종 결정권자 대신 후보 평가기로 쓰는 비교 대상. |
| E4 | [NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models (2024)](https://arxiv.org/abs/2407.12366) | 시각과 LLM을 정렬하고 navigation policy network로 행동 예측 | 설명 가능한 추론과 전문 정책의 결합. 임무 할당이나 SAC 계층과 정확히 같은 구조는 아님. |
| E5 | [SG-Nav: Online 3D Scene Graph Prompting for LLM-based Zero-shot Object Navigation (2024)](https://arxiv.org/abs/2410.08189) | 객체·그룹·방의 graph로 목표 위치 추론, 재지각으로 오류 수정 | VLM/LLM 입력을 구조화하고 기억하는 방식의 참고. 군중 속도·의도는 별도 표현이 필요. |
| E6 | [Mobility VLA: Multimodal Instruction Navigation with Long-Context VLMs and Topological Graphs (2024)](https://arxiv.org/abs/2407.07775) | 영상 tour와 지시로 목표 프레임 선정 → topological navigation 정책 | 장기 시각 기억과 장소 검색. 매 시점 속도 명령을 VLM이 직접 생성하는 구조와 다름. |

## 7. F — 시각·언어에서 행동을 직접 만드는 VLA (3편)

| ID | 연구·출처 | 행동 생성 | 적용 및 주의점 |
| --- | --- | --- | --- |
| F1 | [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control (2023)](https://arxiv.org/abs/2307.15818) | 로봇 행동을 토큰화하여 VLM과 로봇 궤적을 공동 fine-tuning | 직접 행동 의사결정의 대표 선례. 통상적인 ‘상위 VLM → 별도 SAC’와 다름. |
| F2 | [OpenVLA: An Open-Source Vision-Language-Action Model (2024)](https://arxiv.org/abs/2406.09246) | 시각·언어에서 행동을 예측하는 공개 VLA, 시연 기반 적응 | 행동 모델 재학습의 공개 출발점. 원 모델을 RL로 학습했다고 설명하지 않음. |
| F3 | [π0: A Vision-Language-Action Flow Model for General Robot Control (2024)](https://arxiv.org/abs/2410.24164) | VLM 표현과 flow-matching 행동 생성 | 연속 행동의 대안. 군중 대피용 action space·데이터를 별도 마련해야 함. |

## 8. G — 모델이 행동 대신 RL 학습 신호를 만드는 연구 (5편)

| ID | 연구·출처 | VLM/LLM 역할 | 적용 및 주의점 |
| --- | --- | --- | --- |
| G1 | [RL-VLM-F: Reinforcement Learning from Vision Language Foundation Model Feedback (2024)](https://arxiv.org/abs/2402.03681) | 영상 관측 쌍의 선호 판단으로 reward model 학습, RL 수행 | 상태를 보고 임무를 배정하는 것과 다름. 정확한 위험 구역 인원·시간을 이미 계산할 수 있다면 VLM 보상을 대체물로 도입할 이점부터 검증해야 함. |
| G2 | [RoboCLIP: One Demonstration is Enough to Learn Robot Policies (2023)](https://arxiv.org/abs/2310.07899) | video-language embedding의 궤적–텍스트/영상 유사도를 보상으로 사용 | 보상 없는 과제에 참고. crowd heatmap 렌더링에서의 유사도 유효성은 검증되지 않음. |
| G3 | [Eureka: Human-Level Reward Design via Coding Large Language Models (2023)](https://arxiv.org/abs/2310.12931) | LLM 보상 코드 생성 → RL 학습 → 성과 피드백으로 코드 개선 | 역할별 보상 설계의 대안. 실행 시 상위 플래너가 아님. |
| G4 | [DrEureka: Language Model Guided Sim-To-Real Transfer (2024)](https://arxiv.org/abs/2406.01967) | 보상과 domain randomization 분포를 LLM으로 설계 | 시뮬레이션 설정 개선에 참고. 임의 군중 모델을 실제 사람에 맞게 보정해 주는 방법은 아님. |
| G5 | [Text2Reward: Reward Shaping with Language Models for Reinforcement Learning (2023)](https://arxiv.org/abs/2309.11489) | 자연어 목표에서 dense reward 프로그램 생성·정제 | 임무별 학습 보상 정의의 참고. 최종 대피 성과 지표와 분리해 reward hacking을 확인해야 함. |

## 9. H — 사람과 공존하는 이동 및 상호작용 (4편)

| ID | 연구·출처 | 핵심 구조 | 적용 및 주의점 |
| --- | --- | --- | --- |
| H1 | [VLM-Social-Nav: Socially Aware Robot Navigation through Scoring using Vision-Language Models (2024)](https://arxiv.org/abs/2404.00210) | VLM이 사회적 상황의 행동 비용을 산출, underlying planner와 결합 | 군중 상황을 점수로 번역하는 참고. 사람 사이를 잘 통과하는 것과 사람을 대피시키는 것은 목적이 다름. |
| H2 | [HSAC-LLM: Enhancing Socially-Aware Robot Navigation through Bidirectional Natural Language Conversation (2024)](https://arxiv.org/abs/2409.04965) | Hybrid SAC의 연속·이산 행동과 LLM의 사람–로봇 양방향 대화 결합 | 현재 혼합 행동 SAC 및 신호 채널과 연관성이 큼. VLM 기반 다중 로봇 임무 배정 논문은 아님. |
| H3 | [Social-LLaVA: Enhancing Robot Navigation through Human-Language Reasoning in Social Spaces (2024 제출/2025 식별자)](https://arxiv.org/abs/2501.09024) | 실제 공공장소 상호작용의 VQA로 사회적 해석·추론·행동 설명 학습 | 관측에서 어떤 사회적 정보를 추출할지 참고. VQA 품질을 대피 성과와 동일시하지 않음. |
| H4 | [AHead-VLN: A Hierarchical Reasoning Framework Guided by Foresight for Human-Aware Navigation (IJCAI 2026)](https://www.ijcai.org/proceedings/2026/696) | 경로별 미래 위험을 구조화, Brain LLM의 경로 판단과 Cerebellum LLM의 실행 스크립트 분리 | 동적 군중의 위험 예측을 상위 입력에 정렬하는 참고. 하위가 RL이라고 주장하지 않음. |

## 10. I — 자율주행에서의 판단·계획·제어 (3편)

| ID | 연구·출처 | 핵심 구조 | 적용 및 주의점 |
| --- | --- | --- | --- |
| I1 | [DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models (2024)](https://arxiv.org/abs/2402.12289) | 장면 설명·분석·계층 계획, DriveVLM-Dual은 기존 주행 스택과 결합 | 느린 의미 판단과 기존 빠른 주행 계층을 연결하는 선례. |
| I2 | [LMDrive: Closed-Loop End-to-End Driving with Large Language Models (2023)](https://arxiv.org/abs/2312.07488) | multimodal 센서·언어 지시로 closed-loop 주행 | 동적 장면에서 언어 지시를 실행하는 대안. 상위 LLM–하위 RL 분리와 동일하지 않음. |
| I3 | [DiLu: A Knowledge-Driven Approach to Autonomous Driving with Large Language Models (2023)](https://arxiv.org/abs/2309.16292) | LLM reasoning·reflection·경험 기억으로 운전 판단 | 실패 경험 회수와 판단의 비교 대상. 이미지 입력 VLM으로 분류하지 않음. |

## 11. 군중 대피라는 이유만으로 VLM 선행연구에 포함하면 안 되는 사례

[RESCUE: Crowd Evacuation Simulation via Controlling SDM-United Characters (ICCV 2025)](https://arxiv.org/abs/2507.20117)는 3D-adaptive SFM 의사결정과 개인화된 물리적 보행 제어를 결합하는 군중 대피 시뮬레이션이다. 이 논문을 **VLM이 대피 로봇 임무를 배정한 연구**라고 인용하면 안 된다. 군중 움직임·물리 모델의 별도 관련 연구로는 참고할 수 있다.

## 12. 이 프로젝트에서 우선 읽을 연구

1. **NaVILA(A2):** 하위 RL에 상위 명령을 전달하는 분리와 명령 실행 계약.
2. **SayCan(A1):** 의미상 적합성과 학습된 실행 가능성의 결합.
3. **COHERENT(C4), LLaMAR(C6):** 중앙 배정과 다중 로봇 실행 피드백·재할당.
4. **Hi Robot(B1):** 현재 시각 상황·피드백에 맞춘 하위 지시 생성.
5. **RoBridge(A4):** 자유 문장 대신 구조화된 실행 표현.
6. **CoINS(A7):** 로봇 스킬의 실제 한계와 인과적 개입 필요성을 상위 판단에 넣기.
7. **What Matters in Orchestrating Robot Policies(B7):** 종료·전환 조건, 기억·관측 표현의 체계적인 비교.
8. **VLFM(A3), HSAC-LLM(H2):** 각각 VLM 목표 선택–RL 이동과 혼합 행동·사람 상호작용의 구체적 참고.

## 13. 적용 방향 — 논문에서 확인한 사실과 구분한 제안

아래는 이 프로젝트를 위한 **설계 제안**이며 위 논문의 군중 대피 검증 결과가 아니다.

### 상위와 하위 사이의 계약

상위가 ‘유입 차단’ 같은 문장만 전달해서는 하위 정책의 목표가 정의되지 않는다. 예를 들면 다음 필드를 갖는 제한된 임무 schema를 둔다.

```json
{
  "robot_id": 0,
  "mission": "guide_out",
  "target_region_id": "crowd_cluster_2",
  "goal_region_id": "safe_region_north",
  "constraints": {"avoid_region_ids": ["hazard_1"]},
  "termination": {"type": "cleared_or_timeout", "timeout_s": 30.0}
}
```

- `guide_out`, `block_inflow`, `reposition`, `hold` 등 실제로 학습·실행할 수 있는 임무만 허용한다. 목적·제약 영역은 관측 및 지도에서 얻은 유효 ID로 제한한다. 이 목록·30초 timeout은 예시이며 합의된 실험 기본값이 아니다.
- 하위는 `π(a | observation, mission, target, goal, constraints)` 형태의 mission-conditioned 정책 또는 임무별 스킬 라이브러리로 준비한다. 기존 단일 목표 SAC가 새 역할을 알아서 수행한다고 가정하지 않는다.
- 사람의 인지·순응 성향 등 숨은 시뮬레이터 상태를 상위에 제공하지 않는다. 숫자로 계산 가능한 인원·속도·유입률은 관측 처리로 계산하고, VLM이 이미지에서 모두 추측하게 하지 않는다.
- ‘유입 차단’이 안내·경고인지 몸체로 통로를 막는 것인지 명세해야 한다. 양방향 통로에서 대피 유출까지 방해하는 행동은 성공적인 차단으로 보상하면 안 된다.

### 실행 가능성과 전환

- 위치·경로·임무별 정책의 가능 범위로 후보를 먼저 검사한다. 상위가 고른 임무가 불가능하면 피드백을 주고 재배정한다.
- SayCan의 성공 value와 현재 SAC의 soft Q를 그대로 같은 것으로 쓰지 않는다. SAC Q는 누적 보상·entropy·할인·스케일에 좌우되며 성공 확률이 아니다. 성공/시간 예측기를 별도로 보정하거나, 같은 조건에서 비교 가능한 임무 가치 모델을 학습한다.
- 상위는 하위보다 느리게 판단하되 고정 주기만으로 운용하지 않는다. 완료·막힘·유입 증가·위험 변화 등 사건으로 재계획하고, 최소 유지 시간/전환 비용/긴급 취소 조건을 따로 정의한다. 의미 없는 역할 진동을 측정한다.
- VLM 지연을 무시한 일시정지 시뮬레이션 평가와 실제 비동기 실행을 구분한다. 추론 중 군중이 이동하므로 관측 timestamp, 명령 유효기간 및 실행 실패 처리가 필요하다.

### VLM이 필요한지 확인하는 실험

하위 실행기를 동일하게 두고 상위만 바꾸는 비교가 우선이다.

| 상위 방식 | 비교 이유 |
| --- | --- |
| 고정 임무 배정·규칙 기반 할당 | 단순 분업 효과 확인 |
| 거리·위험·유입량 비용의 최적화 할당 | 언어 모델 없이도 풀리는 조합 최적화인지 확인 |
| 학습된 상위 RL/HRL | 같은 임무 집합에서 모델 기반 상식의 이득 확인 |
| 구조화 상태를 받는 텍스트 LLM | VLM의 시각 입력 자체가 필요한지 분리 |
| 이미지+구조화 상태를 받는 VLM | 시각·언어 추론의 추가 이득 확인 |
| VLM+실행 가능성 검사·피드백 | grounding과 재계획 기여 확인 |

전체 정책의 평가는 기존 신호 OFF·scripted shuttle 대조도 유지한다. 새로운 지도·군중 유입·체류·위험 변화·운영 지시에 대한 일반화, 위험 구역 누적 인원·시간, 재진입, 차단으로 인한 유출 방해, 임무 전환 횟수, 추론 비용·지연을 함께 본다.

**[Likely]** 모든 관측과 목표가 고정된 숫자 상태로 주어지고 가능한 임무가 몇 개뿐이면 규칙/최적화/상위 RL도 강한 기준선일 수 있다. 반면 언어로 바뀌는 운영 제약이나 미지의 의미적 상황에 적응하는 것이 목표라면 VLM의 필요성을 더 설득력 있게 검증할 수 있다. 이 판단은 실험으로 확인해야 한다.
