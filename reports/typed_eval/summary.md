# Typed-path evaluation

80 held-out test cases, 10 per class, half written in Thai and half in English. Gemini wrote each case as a chat message without its disaster type; the Gemini agent read it back and called the ML models, exactly as in the app. The ML models make every prediction.

## Direct features vs typed message (same cases)

| Model | Direct accuracy | Typed accuracy | Direct Macro-F1 | Typed Macro-F1 | Macro-F1 change |
|---|---|---|---|---|---|
| Random Forest | 91.2% | 86.2% | 0.913 | 0.865 | -0.048 |
| LightGBM | 93.8% | 88.8% | 0.937 | 0.888 | -0.049 |
| XGBoost | 90.0% | 82.5% | 0.899 | 0.828 | -0.071 |

## The LLM as interpreter

- Ran the models via classify_event: **80/80** (100%)
- All 6 attributes read correctly: **68/80**
- Attributes read correctly overall: **96.7%**
- Explanations matching the model output: **78/80**
- Gemini requests per case: 3.1 on average (writer + agent rounds + fallbacks)

| Attribute | Read correctly |
|---|---|
| Country | 91% |
| Region | 96% |
| Start month | 98% |
| Event duration | 99% |
| Magnitude scale | 96% |
| Magnitude value | 100% |

## Per class (Random Forest)

| Class | Cases | Direct correct | Typed correct |
|---|---|---|---|
| Drought | 10 | 8/10 | 8/10 |
| Earthquake | 10 | 10/10 | 9/10 |
| Epidemic | 10 | 9/10 | 7/10 |
| Extreme Temperature | 10 | 10/10 | 9/10 |
| Flood | 10 | 9/10 | 8/10 |
| Landslide | 10 | 9/10 | 9/10 |
| Storm | 10 | 10/10 | 10/10 |
| Wildfire | 10 | 8/10 | 9/10 |

## Cases right on direct features but wrong when typed (Random Forest): 5

- CASE-2781 (Earthquake → Epidemic), misread: Magnitude scale · “เกิดเหตุที่แอลจีเรียในทวีปแอฟริกาเมื่อเดือนกุมภาพันธ์ที่ผ่านมา มีการวัดขนาดเหตุการณ์นี้แต่ไม่มีการบันทึกตัวเลขไว้”
- CASE-0030 (Epidemic → Landslide), misread: nothing · “In Congo, Africa, something happened in September lasting less than a day.”
- CASE-2333 (Epidemic → Landslide), misread: Magnitude scale · “In Indonesia, a vaccination drive was reported in May.”
- CASE-1799 (Extreme Temperature → Landslide), misread: Magnitude scale · “In Argentina, temperatures were recently reported for January without any specific figures.”
- CASE-1484 (Flood → Wildfire), misread: Country, Region · “เกิดเหตุในฮ่องกง ประเทศจีน ช่วงเดือนกรกฎาคม โดยมีการวัดพื้นที่ได้รับผลกระทบแต่ไม่ได้ระบุตัวเลข”

Small sample: treat differences of a few cases as noise. The full-test-set figures (3,495 cases) on the Leaderboard tab remain the reference for the models themselves.
