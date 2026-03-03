from config.tsukuba_conf.MMK_Tsukuba_96for96_weather_timeF import exp_conf as _base
exp_conf = _base.copy()
exp_conf["exp_name"]    = "MMK_Tsukuba_96for96_weather_timeF_v2"
exp_conf["learning_rate"]= 5e-4
exp_conf["max_epochs"]   = max(_base.get("max_epochs",20), 50)
exp_conf["es_patience"]  = max(_base.get("es_patience",5), 10)
exp_conf["criterion"]    = "mse"; exp_conf["loss"] = "mse"
if "weight_decay" in exp_conf: exp_conf["weight_decay"] = 0.0
if "dropout" in exp_conf:      exp_conf["dropout"] = 0.1
# （MMKの容量を少し増やせるキーがあれば併せて上げる：例）
for k in ["n_experts","hidden_size","d_model","width"]:
    if k in exp_conf: exp_conf[k] = int(exp_conf[k])*2 if isinstance(exp_conf[k],int) and exp_conf[k]>0 else exp_conf[k]
