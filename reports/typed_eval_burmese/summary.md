# Typed-path evaluation

40 held-out test cases, 5 per class, all written in Burmese. Gemini wrote each case as a chat message without its disaster type; the Gemini agent read it back and called the ML models, exactly as in the app. The ML models make every prediction.

## Direct features vs typed message (same cases)

| Model | Direct accuracy | Typed accuracy | Direct Macro-F1 | Typed Macro-F1 | Macro-F1 change |
|---|---|---|---|---|---|
| Random Forest | 87.5% | 72.5% | 0.866 | 0.702 | -0.164 |
| LightGBM | 87.5% | 72.5% | 0.866 | 0.688 | -0.178 |
| XGBoost | 87.5% | 70.0% | 0.874 | 0.678 | -0.196 |

## The LLM as interpreter

- Ran the models via classify_event: **40/40** (100%)
- All 6 attributes read correctly: **24/40**
- Attributes read correctly overall: **92.5%**
- Asked the user for more details first (too little given): **36/40**
- Of those, asked to confirm a scale from what the message described: **24** (Km2, Kph, Moment Magnitude, Vaccinated, °C); confirmed and used: **18**
- Explanations matching the model output: **39/39**
- Gemini requests per case: 5.1 on average (writer + agent rounds + fallbacks)

| Attribute | Read correctly |
|---|---|
| Country | 100% |
| Region | 100% |
| Start month | 92% |
| Event duration | 100% |
| Magnitude scale | 70% |
| Magnitude value | 92% |

## Per class (Random Forest)

| Class | Cases | Direct correct | Typed correct |
|---|---|---|---|
| Drought | 5 | 4/5 | 4/5 |
| Earthquake | 5 | 5/5 | 3/5 |
| Epidemic | 5 | 4/5 | 2/5 |
| Extreme Temperature | 5 | 5/5 | 4/5 |
| Flood | 5 | 5/5 | 5/5 |
| Landslide | 5 | 5/5 | 5/5 |
| Storm | 5 | 5/5 | 5/5 |
| Wildfire | 5 | 2/5 | 1/5 |

## Cases right on direct features but wrong when typed (Random Forest): 6

- CASE-0527 (Earthquake → Flood), misread: Magnitude scale · “မလေးရှားနိုင်ငံမှာ မဂ္ဂနီကျူ့ ၉.၁ ရှိတဲ့ ပြင်းအားတစ်ခု ဒီဇင်ဘာလထဲမှာ ခဏလေး ဖြစ်သွားတယ်။”
- CASE-3008 (Earthquake → Drought), misread: Magnitude scale, Magnitude value · “တရုတ်နိုင်ငံမှာ ဇန်နဝါရီလအတွင်းက ပြင်းအား ၅.၉ ရှိတဲ့ ဖြစ်စဉ်တစ်ခု ဖြစ်ပွားခဲ့ပါတယ်။”
- CASE-2333 (Epidemic → Flood), misread: Magnitude scale · “အင်ဒိုနီးရှားနိုင်ငံမှာ မေလအတွင်းက လူဦးရေဘယ်လောက်ကို ကာကွယ်ဆေးထိုးပေးခဲ့တယ်ဆိုတာကို မှတ်တမ်းတင်ထားပါတယ်။”
- CASE-1394 (Epidemic → Drought), misread: Magnitude scale · “ယီမင်နိုင်ငံမှာ ဇန်နဝါရီလကစပြီး ၃၃၄ ရက်ကြာခဲ့ပြီး ကာကွယ်ဆေးထိုးနှံထားသူ အရေအတွက်ကို မှတ်တမ်းတင်ထားပါတယ်။”
- CASE-3433 (Extreme Temperature → Storm), misread: Magnitude scale · “ဟန်ဂေရီနိုင်ငံမှာ ဖေဖော်ဝါရီလအတွင်းက အပူချိန်တွေကို တိုင်းတာမှတ်တမ်းတင်ခဲ့ပါတယ်။”
- CASE-1311 (Wildfire → Storm), misread: Magnitude scale, Magnitude value · “စပိန်နိုင်ငံမှာ ဇန်နဝါရီလကစပြီး ရက်ပေါင်း ၂၁၂ ရက်ကြာခဲ့တဲ့အပြင် ၁၁၉၀ စတုရန်းကီလိုမီတာလောက်ထိ ထိခိုက်မှုတွေ ဖြစ်ပေါ်ခဲ့ပါတယ်။”

Small sample: treat differences of a few cases as noise. The full-test-set figures (3,495 cases) on the Leaderboard tab remain the reference for the models themselves.
