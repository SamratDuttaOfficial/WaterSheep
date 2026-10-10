{{header}}

{{title}} answers yes/no, single-choice, rating and multi-label questions about any text, with a
probability for every option. Version {{version}} (`{{name}}`).

## Usage

```bash
pip install transformers torch
```

```python
from transformers import pipeline

ws = pipeline(model="{{repo_id}}", trust_remote_code=True)
ws("I was charged twice.", question="Which team should handle this?", options=["billing", "shipping", "support"])
```

| Type | Options | Answer |
|---|---|---|
| `noul` | none (yes/no) | probability of yes |
| `choice` | any labels | the best option |
| `score` | a digit scale, e.g. `1` to `5` | the expected level |
| `multi` | any labels, with `type="multi"` | every option above the threshold |

Every answer includes a probability for each option.

## Using Jev?

WaterSheep is an open-source alternative to Jev. Run it as a local server:

```bash
pip install git+https://github.com/SamratDuttaOfficial/WaterSheep
watersheep --model {{repo_id}} --serve
```

It answers Jev's `POST /v1/systemone` requests on your machine, and TypeSafe's Python SDK works against it
without code changes:

```bash
export TYPESAFE_BASE_URL=http://127.0.0.1:8766
```

Any API key value works locally. Multi-label questions (`"type": "multi"`) work too, as plain JSON.
WaterSheep is independent and not affiliated with TypeSafe AI.

## Download

```bash
hf download {{repo_id}} --local-dir {{title}}
```

Or with Git (requires Git LFS):

```bash
git clone https://huggingface.co/{{repo_id}}
```

Then load it from the folder, offline:

```python
ws = pipeline(model="{{title}}", trust_remote_code=True)
```

## API

Deploy it as an [Inference Endpoint](https://endpoints.huggingface.co), then:

```bash
curl https://YOUR-ENDPOINT -H "Authorization: Bearer $HF_TOKEN" -H "Content-Type: application/json" -d '{"inputs": "I was charged twice.", "parameters": {"question": "Which team should handle this?", "options": ["billing", "shipping", "support"]}}'
```

{{javascript}}

## Evaluation

{{metrics}}

ECE is the expected calibration error (lower is better).

{{calibration}}

{{benchmarks}}

{{training}}

## Limitations

- English only.
- Long inputs are truncated.
- Rating-scale answers are less accurate than the other types.
- Probabilities are calibrated on data like the training data; validate them on your own.
- Not for high-stakes decisions (medical, legal, financial, hiring) on its own.

## License

Apache 2.0 (`LICENSE`). Trained on openly licensed data; credits in `NOTICE`.

## Citation

```bibtex
@misc{watersheep,
  author = {{{author}}},
  title  = {{{title}}: calibrated decisions for any text},
  year   = {{{year}}},
  url    = {https://huggingface.co/{{repo_id}}}
}
```

---

<p align="center">
  <a href="https://doi.org/10.13140/RG.2.2.28606.45122"><img src="https://samratduttaofficial.github.io/WaterSheep/preprint.svg" alt="Read the preprint" width="400"></a>
  <a href="https://buymeacoffee.com/samratdutta"><img src="https://samratduttaofficial.github.io/WaterSheep/coffee.svg" alt="Buy me a coffee" width="400"></a>
</p>
