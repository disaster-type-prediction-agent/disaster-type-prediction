# Typed-path evaluation

40 held-out test cases, 5 per class, all written in Burmese. Gemini wrote each case as a chat message without its disaster type; the Gemini agent read it back and called the ML models, exactly as in the app. The ML models make every prediction.

## Direct features vs typed message (same cases)

| Model | Direct accuracy | Typed accuracy | Direct Macro-F1 | Typed Macro-F1 | Macro-F1 change |
|---|---|---|---|---|---|
| Random Forest | 87.5% | 57.5% | 0.866 | 0.530 | -0.336 |
| LightGBM | 87.5% | 60.0% | 0.866 | 0.551 | -0.316 |
| XGBoost | 87.5% | 57.5% | 0.874 | 0.543 | -0.331 |

## The LLM as interpreter

- Ran the models via classify_event: **39/40** (98%)
- All 6 attributes read correctly: **23/39**
- Attributes read correctly overall: **92.7%**
- Asked the user for more details first (too little given): **37/40**
- Of those, asked to confirm a scale from what the message described: **26** (Km2, Kph, Moment Magnitude, Vaccinated, °C); confirmed and used: **18**
- Explanations matching the model output: **36/36**
- Gemini requests per case: 8.9 on average (writer + agent rounds + fallbacks)

| Attribute | Read correctly |
|---|---|
| Country | 97% |
| Region | 97% |
| Start month | 92% |
| Event duration | 100% |
| Magnitude scale | 72% |
| Magnitude value | 97% |

## Per class (Random Forest)

| Class | Cases | Direct correct | Typed correct |
|---|---|---|---|
| Drought | 5 | 4/5 | 3/5 |
| Earthquake | 5 | 5/5 | 5/5 |
| Epidemic | 5 | 4/5 | 1/5 |
| Extreme Temperature | 5 | 5/5 | 4/5 |
| Flood | 5 | 5/5 | 5/5 |
| Landslide | 5 | 5/5 | 0/5 |
| Storm | 5 | 5/5 | 4/5 |
| Wildfire | 5 | 2/5 | 1/5 |

## Cases right on direct features but wrong when typed (Random Forest): 11

- CASE-1430 (Drought → Wildfire), misread: Country, Region · “ဇွန်လတုန်းက ကာဘိုဗာဒီနိုင်ငံမှာ ထိခိုက်မှုဖြစ်ခဲ့တဲ့ဧရိယာကို တိုင်းတာထားပြီးပါပြီ။”
- CASE-2333 (Epidemic → Flood), misread: Magnitude scale · “အင်ဒိုနီးရှားနိုင်ငံမှာ မေလအတွင်းက လူဦးရေဘယ်လောက်ကို ကာကွယ်ဆေးထိုးပေးခဲ့တယ်ဆိုတာကို မှတ်တမ်းတင်ထားပါတယ်။”
- CASE-0165 (Epidemic → Flood), misread: nothing · “ဂီနီနိုင်ငံမှာ စက်တင်ဘာလအတွင်းက တစ်ရက်တည်းနဲ့ ဖြစ်ရပ်တစ်ခု ဖြစ်ပွားခဲ့ပါတယ်။”
- CASE-1394 (Epidemic → Drought), misread: Magnitude scale · “ယီမင်နိုင်ငံမှာ ဇန်နဝါရီလကစပြီး ၃၃၄ ရက်ကြာခဲ့ပြီး ကာကွယ်ဆေးထိုးနှံထားသူ အရေအတွက်ကို မှတ်တမ်းတင်ထားပါတယ်။”
- CASE-3433 (Extreme Temperature → Storm), misread: Magnitude scale · “ဟန်ဂေရီနိုင်ငံမှာ ဖေဖော်ဝါရီလအတွင်းက အပူချိန်တွေကို တိုင်းတာမှတ်တမ်းတင်ခဲ့ပါတယ်။”
- CASE-1245 (Landslide → Storm), misread: nothing · “ဂျပန်နိုင်ငံမှာ ဇူလိုင်လအတွင်းက ဖြစ်ပျက်ခဲ့တဲ့ အခြေအနေတစ်ခုပါ။”
- CASE-0029 (Landslide → Flood), misread: nothing · “အင်ဒိုနီးရှားနိုင်ငံ အာရှဒေသမှာ ဧပြီလအတွင်းက ဖြစ်ပျက်ခဲ့တာမျိုးရှိပါတယ်။”
- CASE-0735 (Landslide → Flood), misread: nothing · “အိန္ဒိယနိုင်ငံမှာ ဩဂုတ်လအတွင်းက ရုတ်တရက်ဖြစ်ရပ်တစ်ခု ဖြစ်ပွားခဲ့ပါတယ်။”
- CASE-3413 (Landslide → Storm), misread: nothing · “ဖိလစ်ပိုင်နိုင်ငံမှာ ဧပြီလအတွင်းက ဖြစ်ပျက်ခဲ့တာ တစ်ရက်တောင် မကြာလိုက်ပါဘူး။”
- CASE-2119 (Landslide → Flood), misread: nothing · “အင်ဒိုနီးရှားနိုင်ငံမှာ မတ်လအတွင်းက ဖြစ်ပျက်ခဲ့တဲ့ အခြေအနေတစ်ခုပါ။”
- CASE-1311 (Wildfire → Storm), misread: Magnitude scale · “စပိန်နိုင်ငံမှာ ဇန်နဝါရီလကစပြီး ရက်ပေါင်း ၂၁၂ ရက်ကြာခဲ့တဲ့အပြင် ၁၁၉၀ စတုရန်းကီလိုမီတာလောက်ထိ ထိခိုက်မှုတွေ ဖြစ်ပေါ်ခဲ့ပါတယ်။”

Small sample: treat differences of a few cases as noise. The full-test-set figures (3,495 cases) on the Leaderboard tab remain the reference for the models themselves.
