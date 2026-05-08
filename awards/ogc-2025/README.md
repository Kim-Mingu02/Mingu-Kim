<div align="center">

# OGC 2025 · 최적화 그랜드 챌린지

**도전상 수상 · 예공제미남들 팀장**  
그래프 기반 적재·하역·재취급 최적화 알고리즘

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square)](#source-code--files)
[![Award](https://img.shields.io/badge/Award-도전상-0064FF?style=flat-square)](#award-record)
[![Feasible](https://img.shields.io/badge/Feasible-True-3182F6?style=flat-square)](results/results_prob5.json)
[![Objective](https://img.shields.io/badge/Objective-5843.0-4E86FF?style=flat-square)](results/results_prob5.json)

</div>

---

## Source Code & Files

OGC 2025 제출 및 실험에 사용한 코드, 문제 파일, 결과 파일, 시각화 파일은 이 페이지 상단에 바로 접근할 수 있도록 정리했습니다.

| 구분 | 링크 | 설명 |
|---|---|---|
| Algorithm Strategy | [algorithm-strategy.md](algorithm-strategy.md) | 선박 수요 배치 최적화 휴리스틱을 단계별로 설명한 문서입니다. |
| Algorithm Code | [myalgorithm_final.py](src/myalgorithm_final.py) · [myalgorithm_variant.py](src/myalgorithm_variant.py) | 업로드한 알고리즘 코드 파일입니다. |
| Utility Code | [util.py](src/util.py) | 실행가능성 검증(feasibility check), BFS, Dijkstra, path backtracking 유틸리티입니다. |
| Visualization Code | [visual_grid.py](src/visual_grid.py) · [visual_results.py](src/visual_results.py) | Grid graph 및 결과 상태를 Plotly 기반 HTML로 시각화하는 코드입니다. |
| Test Data | [test_prob1.json](data/test_prob1.json) · [test_prob2.json](data/test_prob2.json) · [test_prob3.json](data/test_prob3.json) · [test_prob4.json](data/test_prob4.json) · [test_prob5.json](data/test_prob5.json) | OGC 문제 인스턴스(input instances)입니다. |
| Result | [results_prob5.json](results/results_prob5.json) | 업로드한 결과 JSON 파일입니다. |
| HTML Visualization | [prob1.html](visualizations/prob1.html) · [prob2.html](visualizations/prob2.html) · [prob3.html](visualizations/prob3.html) · [prob4.html](visualizations/prob4.html) · [prob5.html](visualizations/prob5.html) | 사전 생성된 3D 시각화 HTML 파일입니다. |
| Manifest | [MANIFEST.md](MANIFEST.md) | 원본 업로드 파일명과 repository 저장 경로를 정리한 파일입니다. |

> GitHub에서 `.html` 파일을 클릭하면 인터랙티브 화면이 아니라 HTML source code로 보일 수 있습니다. 시각화를 직접 조작하려면 HTML 파일을 로컬 브라우저에서 열거나 GitHub Pages로 배포하면 됩니다.

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
| 참가 규모 | 총 343개 팀 참가, 상장 기재 기준 |
| 상장 발급 표기 | LG CNS · 대한산업공학회 |
| 비고 | 시상식 사진의 배경 화면에는 `OPTIMIZATION GRAND CHALLENGE 2024` 문구가 보입니다. README의 수상 연도와 수상명은 업로드한 상장 기준으로 `OGC 2025`와 `도전상`으로 정리했습니다. |

<div align="center">

### 상장

<img src="images/ogc-2025-certificate.png" width="560" alt="OGC 2025 도전상 상장 사진">

### 시상식 참여

<img src="images/ogc-2025-award-ceremony.png" width="780" alt="OGC 시상식 참여 사진">

</div>

---

## Result Summary

업로드된 `results_prob5.json` 기준 결과 요약입니다. 아래 값은 README에서 재계산한 값이 아니라, 업로드된 결과 파일에 기록된 값을 정리한 것입니다.

| Metric | Value |
|---|---:|
| Problem | `prob5` |
| Feasible | `True` |
| Objective value | `5843.0` |
| Runtime | `57.62 sec` |
| Timelimit exception | `False` |
| Total routes | `1015` |

---

## Problem Instances

| Instance | Data | Visualization | N | Edges | Ports | Demand types | F | LB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `prob1` | [JSON](data/test_prob1.json) | [HTML](visualizations/prob1.html) | 77 | 106 | 7 | 13 | 100 | 28117 |
| `prob2` | [JSON](data/test_prob2.json) | [HTML](visualizations/prob2.html) | 89 | 122 | 10 | 23 | 100 | 32138 |
| `prob3` | [JSON](data/test_prob3.json) | [HTML](visualizations/prob3.html) | 85 | 113 | 10 | 13 | 100 | 37882 |
| `prob4` | [JSON](data/test_prob4.json) | [HTML](visualizations/prob4.html) | 162 | 225 | 12 | 34 | 100 | 77361 |
| `prob5` | [JSON](data/test_prob5.json) | [HTML](visualizations/prob5.html) | 279 | 399 | 10 | 25 | 100 | 107637 |

---

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
