<div align="center">

# OGC 2025 · 최적화 그랜드 챌린지

**도전상 수상 · 예공제미남들 팀장**  
그래프 기반 적재·하역·재취급 최적화 알고리즘

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square)](#source-code--files)
[![Award](https://img.shields.io/badge/Award-도전상-0064FF?style=flat-square)](#award-record)

</div>

---


## Award Record

2025년 **최적화 그랜드 챌린지 2025(Optimization Grand Challenge 2025, OGC 2025)**에 **예공제미남들** 팀으로 참가하여 **도전상**을 수상했습니다. 팀장으로 참여했으며, 시상식에 참석했습니다.

| 항목 | 내용 |
|---|---|
| 대회 | 최적화 그랜드 챌린지 2025 / OGC 2025 |
| 수상 | 도전상 |
| 팀명 | 예공제미남들 |
| 역할 | 팀장 |
| 시상일 | 2025년 9월 19일 |
| 참가 규모 | 총 343개 팀 참가 |



## Algorithm Notes

| Component | Main files / functions | 내용 |
|---|---|---|
| Graph Search | `PathFinder`, `find_path`, `k_shortest_paths_cached` | 점유 노드(occupied nodes)를 고려한 경로 탐색 및 shortest path 계산입니다. |
| Loading Heuristic | `loading_heuristic_farthest_first`, `loading_heuristic_flexible_score` | 적재 위치를 거리, blocking risk, passageway, 도착지 특성으로 평가합니다. |
| Unloading Heuristic | `unloading_heuristic` | 목적 항목을 하역하면서 corridor blocker와 재취급(rehandling)을 처리합니다. |
| Rehandling Decision | `intelligent_rehandling_decider` | relocation과 temporary unloading/reloading 전략을 비용 기준으로 비교합니다. |
| Feasibility Check | `check_feasibility` | route 구조, edge validity, loading/unloading 상태, demand 충족 여부, objective value를 검증합니다. |
| Visualization | `visualize_grid_graph_final`, `create_visualization` | 3D grid graph와 port별 결과 상태를 HTML로 생성합니다. |

---

## Run Examples

Install dependencies:

```bash
python -m pip install -r awards/ogc-2025/requirements.txt
```

Generate a 3D grid visualization:

```bash
python awards/ogc-2025/src/visual_grid.py awards/ogc-2025/data/test_prob1.json
```

Generate a result-state visualization:

```bash
python awards/ogc-2025/src/visual_results.py   awards/ogc-2025/data/test_prob5.json   awards/ogc-2025/results/results_prob5.json
```

Run the uploaded algorithm module:

```bash
cd awards/ogc-2025/src
python - <<'PYRUN'
import json
from myalgorithm_final import algorithm

with open('../data/test_prob5.json', encoding='utf-8') as f:
    prob_info = json.load(f)

solution = algorithm(prob_info, timelimit=60)
print(type(solution))
PYRUN
```

---

<div align="center">

[← Back to Main Profile](../../README.md)

</div>
