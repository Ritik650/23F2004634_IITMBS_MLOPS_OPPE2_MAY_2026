-- post.lua
-- wrk script that sends a fixed, valid /predict payload on every request.
-- Usage: see stress_test.sh

wrk.method = "POST"
wrk.body = [[
{
  "age": 57,
  "gender": "male",
  "cp": 2,
  "trestbps": 140,
  "chol": 240,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.2,
  "slope": 1,
  "ca": 0,
  "thal": 2
}
]]
wrk.headers["Content-Type"] = "application/json"
