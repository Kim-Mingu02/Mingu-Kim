
# GAVRP Project

차량 수요 배치 및 경로 계획 문제를 대상으로 한 최적화 알고리즘 구현 repository입니다.  
업로드된 Python 코드와 결과 시각화 이미지를 아래에서 확인할 수 있습니다.

---

## Code

| File | Description |
|---|---|
| [`gavrp.py`](./gavrp.py) | VRP문제에 대한 GA 알고리즘 구현 파일입니다. |
| [`gavrp_qgm.py`](./gavrp_qgm.py) | 하이브리드 양자어닐링 QPU 알고리즘 구현 파일입니다. |
| [`gavrp_viz.py`](./gavrp_viz.py) | 결과 시각화를 위한 파일입니다. |

---

## Visualization Results
<table>
  <tr>
    <td width="50%" align="center">
      <img src="./KakaoTalk_20251028_233012016.png" width="100%" alt="Visualization 1">
      <br><sub>D Wave 대쉬보드</sub>
    </td>
    <td width="50%" align="center">
      <img src="./KakaoTalk_20251028_233025628.png" width="100%" alt="Visualization 2">
      <br><sub>D Wave 대쉬보드 </sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="./KakaoTalk_20251028_233056827.png" width="100%" alt="Visualization 3">
      <br><sub>Figure 3</sub>
    </td>
    <td width="50%" align="center">
      <img src="./KakaoTalk_20251028_233137686.png" width="100%" alt="Visualization 4">
      <br><sub>Figure 4</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="./KakaoTalk_20251028_233302061.png" width="100%" alt="Visualization 5">
      <br><sub>Figure 5</sub>
    </td>
    <td width="50%" align="center">
      <img src="./KakaoTalk_20251028_233312866.png" width="100%" alt="Visualization 6">
      <br><sub>Figure 6</sub>
    </td>
  </tr>
</table>

---

## Notes
VRP 문제는 솔로몬 데이터 셋을 사용하였고 양자 어닐링을 사용한 문제가 300 세대를 도달하는것에 있어 2배 가까운 속도를 보여주었고 해의 품질 또한 더 좋음을 확인하였습니다. 
