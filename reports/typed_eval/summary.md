# Typed-path evaluation

80 held-out test cases, 10 per class, half written in Thai and half in English. Gemini wrote each case as a chat message without its disaster type; the Gemini agent read it back and called the ML models, exactly as in the app. The ML models make every prediction.

## Direct features vs typed message (same cases)

| Model | Direct accuracy | Typed accuracy | Direct Macro-F1 | Typed Macro-F1 | Macro-F1 change |
|---|---|---|---|---|---|
| Random Forest | 91.2% | 90.0% | 0.913 | 0.901 | -0.013 |
| LightGBM | 93.8% | 92.5% | 0.937 | 0.924 | -0.013 |
| XGBoost | 90.0% | 91.2% | 0.899 | 0.912 | +0.013 |

## The LLM as interpreter

- Ran the models via classify_event: **80/80** (100%)
- All 6 attributes read correctly: **70/80**
- Attributes read correctly overall: **97.3%**
- Asked the user for more details first (too little given): **17/80**
- Explanations matching the model output: **72/73**
- Gemini requests per case: 5.8 on average (writer + agent rounds + fallbacks)

| Attribute | Read correctly |
|---|---|
| Country | 92% |
| Region | 96% |
| Start month | 95% |
| Event duration | 100% |
| Magnitude scale | 100% |
| Magnitude value | 100% |

## Per class (Random Forest)

| Class | Cases | Direct correct | Typed correct |
|---|---|---|---|
| Drought | 10 | 8/10 | 7/10 |
| Earthquake | 10 | 10/10 | 10/10 |
| Epidemic | 10 | 9/10 | 9/10 |
| Extreme Temperature | 10 | 10/10 | 10/10 |
| Flood | 10 | 9/10 | 8/10 |
| Landslide | 10 | 9/10 | 9/10 |
| Storm | 10 | 10/10 | 10/10 |
| Wildfire | 10 | 8/10 | 9/10 |

## Cases right on direct features but wrong when typed (Random Forest): 2

- CASE-1430 (Drought → Wildfire), misread: Country, Region · “เกิดเหตุการณ์ในภูมิภาคแอฟริกาของประเทศกาบูแวร์ดีเมื่อเดือนมิถุนายนที่ผ่านมา โดยไม่มีการบันทึกขนาดความรุนแรง แต่พื้นที่ได้รับผลกระทบได้รับการรายงานไว้”
- CASE-1484 (Flood → Wildfire), misread: Country, Region · “เกิดเหตุที่ฮ่องกง จีน ในช่วงเดือนกรกฎาคม โดยมีการรายงานพื้นที่ที่ได้รับผลกระทบไว้แล้ว”

Small sample: treat differences of a few cases as noise. The full-test-set figures (3,495 cases) on the Leaderboard tab remain the reference for the models themselves.
