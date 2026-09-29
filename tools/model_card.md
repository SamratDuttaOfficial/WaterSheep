{{header}}

{{title}} answers yes/no, single-choice, rating and multi-label questions about any text, with a
probability for every option. Version {{version}} (`{{name}}`).

## Install

```bash
{{install}}
```

## Usage

The model downloads from Hugging Face on first use, then runs locally.

**Python**

```python
from watersheep import WaterSheep

ws = WaterSheep.load("{{repo_id}}")
ws.decide("I was charged twice.", "Which team should handle this?", ["billing", "shipping", "support"])
```

**Command line**

```bash
watersheep --model {{repo_id}} --question "Which team should handle this?" --options billing,shipping,support --state "I was charged twice."
```

**HTTP API**

```bash
watersheep --model {{repo_id}} --serve
curl http://127.0.0.1:8766/v1/decisions -d '{"state": "I was charged twice.", "questions": {"team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": ["billing", "shipping", "support"]}}}'
```

{{javascript}}

| Type | Question | Answer |
|---|---|---|
| `noul` | yes/no | probability of yes |
| `choice` | single choice | the option, with a probability for each |
| `score` | rating scale | the expected level, with a probability for each |
| `multi` | multi-label | every option above the threshold |

## Download and run locally

Download the model:

```bash
hf download {{repo_id}} --local-dir {{title}}
```

Or `git clone https://huggingface.co/{{repo_id}}` (requires Git LFS).

Then use the `{{title}}` folder in place of the model id:

```python
from watersheep import WaterSheep

ws = WaterSheep.load("{{title}}")
ws.decide("I was charged twice.", "Which team should handle this?", ["billing", "shipping", "support"])
```

```bash
watersheep --model {{title}} --question "Which team should handle this?" --options billing,shipping,support --state "I was charged twice."
watersheep --model {{title}} --serve
```

- **JavaScript:** call `load({ base: "{{title}}/" })` before `decide`, with the folder on your web server.
{{onnx}}

## Evaluation

{{metrics}}

ECE is the expected calibration error (lower is better).

{{benchmarks}}

## Training

- Base model: {{base_model}}, fine-tuned with a decision head.
- Data: openly licensed public datasets (listed in `NOTICE`) and synthetic decisions from Qwen3.5-4B.
- Calibration: a temperature per question type, fitted on a validation split.

## Limitations

- English only.
- Long inputs are truncated.
- Rating-scale answers are less accurate than the other types.
- Probabilities are calibrated on data like the training data; validate them on your own.
- Not for high-stakes decisions (medical, legal, financial, hiring) on its own.

## License

Apache 2.0 (`LICENSE`). Attributions: `NOTICE`.

## Citation

```bibtex
@misc{watersheep,
  author = {{{author}}},
  title  = {{{title}}: calibrated decisions for any text},
  year   = {{{year}}},
  url    = {https://huggingface.co/{{repo_id}}}
}
```
