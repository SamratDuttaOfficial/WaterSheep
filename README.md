![WaterSheep](assets/banner.svg)

# ![](assets/logo.svg) WaterSheep

Calibrated decisions for any text.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-model-yellow)](https://huggingface.co/samratduttaofficial/WaterSheep)
[![Demo](https://img.shields.io/badge/demo-online-brightgreen)](https://huggingface.co/spaces/samratduttaofficial/WaterSheep)

[Website](https://samratduttaofficial.github.io/WaterSheep/) ·
[Demo](https://huggingface.co/spaces/samratduttaofficial/WaterSheep) ·
[Model](https://huggingface.co/samratduttaofficial/WaterSheep)

WaterSheep answers yes/no, single-choice, rating and multi-label questions about any text, with a
probability for every option. The trained model is on Hugging Face, ready to use.

## Install

```bash
pip install git+https://github.com/SamratDuttaOfficial/WaterSheep
```

## Usage

The model downloads from Hugging Face on first use, then runs locally.

**Python**

```python
from watersheep import WaterSheep

ws = WaterSheep.load("samratduttaofficial/WaterSheep")
ws.decide("I was charged twice.", "Which team should handle this?", ["billing", "shipping", "support"])
```

**Command line**

```bash
watersheep --model samratduttaofficial/WaterSheep --question "Which team should handle this?" --options billing,shipping,support --state "I was charged twice."
```

**HTTP API**

```bash
watersheep --model samratduttaofficial/WaterSheep --serve
curl http://127.0.0.1:8766/v1/decisions -d '{"state": "I was charged twice.", "questions": {"team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": ["billing", "shipping", "support"]}}}'
```

**JavaScript**, no install:

```html
<script type="module">
  import { decide } from "https://samratduttaofficial.github.io/WaterSheep/watersheep.js";
  console.log(await decide("I was charged twice.", "Which team should handle this?", ["billing", "shipping", "support"]));
</script>
```

| Type | Question | Answer |
|---|---|---|
| `noul` | yes/no | probability of yes |
| `choice` | single choice | the option, with a probability for each |
| `score` | rating scale | the expected level, with a probability for each |
| `multi` | multi-label | every option above the threshold |

## Download and run locally

Download the model from Hugging Face:

```bash
hf download samratduttaofficial/WaterSheep --local-dir WaterSheep
```

Or `git clone https://huggingface.co/samratduttaofficial/WaterSheep` (requires Git LFS).

Then use the `WaterSheep` folder in place of the model id:

```python
from watersheep import WaterSheep

ws = WaterSheep.load("WaterSheep")
ws.decide("I was charged twice.", "Which team should handle this?", ["billing", "shipping", "support"])
```

```bash
watersheep --model WaterSheep --question "Which team should handle this?" --options billing,shipping,support --state "I was charged twice."
watersheep --model WaterSheep --serve
```

- **JavaScript:** call `load({ base: "WaterSheep/" })` before `decide`, with the folder on your web server.
- **Other languages:** run `WaterSheep/onnx/model_quantized.onnx` with ONNX Runtime.

## Build from source

Only needed to train a new model:

```bash
git clone https://github.com/SamratDuttaOfficial/WaterSheep
cd WaterSheep
./run.sh
```

Use `run.bat` on Windows. `--help` lists all options; `run-benchmarks` evaluates the result.

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
