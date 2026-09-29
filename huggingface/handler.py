from transformers import pipeline


class EndpointHandler:
    def __init__(self, path=""):
        self.pipe = pipeline(model=path, trust_remote_code=True)

    def __call__(self, data):
        return self.pipe(data["inputs"], **(data.get("parameters") or {}))
