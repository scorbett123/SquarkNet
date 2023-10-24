import torch
import io
from model.models import Models

m = torch.load("model_saves/high_qality.saved")
params = m["params"]
model_statedict = torch.load(io.BytesIO(m["models"]))
models = Models(params["n_channels"], params["nbooks"], params["ncodes"], epochs=params["epochs"], device="cpu")

mappings = {
    "discriminator.discrims": "discriminator._discrims",
    "transform.window": "_transform.window",
    ".convs.": "._convs."
}

new_state_dict = {}
for key in model_statedict:
    new_key = key
    if any([i in new_key for i in mappings]):
        for i in mappings:
            new_key = new_key.replace(i, mappings[i])
    new_state_dict[new_key] = model_statedict[key]
models.load_state_dict(new_state_dict)
models.save("model_saves/recovered.saved")