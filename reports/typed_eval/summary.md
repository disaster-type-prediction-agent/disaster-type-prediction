# Typed-path evaluation

80 held-out test cases, 10 per class, half written in Thai and half in English. Gemini wrote each case as a chat message without its disaster type; the Gemini agent read it back and called the ML models, exactly as in the app. The ML models make every prediction.

## Direct features vs typed message (same cases)

| Model | Direct accuracy | Typed accuracy | Direct Macro-F1 | Typed Macro-F1 | Macro-F1 change |
|---|---|---|---|---|---|
| Random Forest | 91.2% | 88.8% | 0.913 | 0.890 | -0.024 |
| LightGBM | 93.8% | 92.5% | 0.937 | 0.924 | -0.013 |
| XGBoost | 90.0% | 87.5% | 0.899 | 0.875 | -0.024 |

## The LLM as interpreter

- Ran the models via classify_event: **80/80** (100%)
- All 6 attributes read correctly: **71/80**
- Attributes read correctly overall: **97.5%**
- Asked the user for more details first (too little given): **21/80**
- Explanations matching the model output: **77/80**
- Gemini requests per case: 5.2 on average (writer + agent rounds + fallbacks)

| Attribute | Read correctly |
|---|---|
| Country | 92% |
| Region | 98% |
| Start month | 98% |
| Event duration | 100% |
| Magnitude scale | 99% |
| Magnitude value | 99% |

## Per class (Random Forest)

| Class | Cases | Direct correct | Typed correct |
|---|---|---|---|
| Drought | 10 | 8/10 | 8/10 |
| Earthquake | 10 | 10/10 | 9/10 |
| Epidemic | 10 | 9/10 | 9/10 |
| Extreme Temperature | 10 | 10/10 | 10/10 |
| Flood | 10 | 9/10 | 8/10 |
| Landslide | 10 | 9/10 | 9/10 |
| Storm | 10 | 10/10 | 10/10 |
| Wildfire | 10 | 8/10 | 8/10 |

## Cases right on direct features but wrong when typed (Random Forest): 2

- CASE-1164 (Earthquake → Epidemic), misread: Magnitude scale · “เกิดเหตุที่เอธิโอเปียในทวีปแอฟริกาเมื่อเดือนมกราคม โดยมีความรุนแรงระดับ 5.2 เป็นเวลา 7 วัน”
- CASE-1484 (Flood → Wildfire), misread: Country, Region · “เกิดเหตุขึ้นที่เขตบริหารพิเศษฮ่องกง ประเทศจีน เมื่อเดือนกรกฎาคม โดยมีการบันทึกพื้นที่ได้รับผลกระทบไว้”

Small sample: treat differences of a few cases as noise. The full-test-set figures (3,495 cases) on the Leaderboard tab remain the reference for the models themselves.
