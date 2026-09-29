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
probability for every option.

## Usage

```bash
pip install transformers torch
```

```python
from transformers import pipeline

ws = pipeline(model="samratduttaofficial/WaterSheep", trust_remote_code=True)
ws("I was charged twice.", question="Which team should handle this?", options=["billing", "shipping", "support"])
```

| Type | Options | Answer |
|---|---|---|
| `noul` | none (yes/no) | probability of yes |
| `choice` | any labels | the best option |
| `score` | a digit scale, e.g. `1` to `5` | the expected level |
| `multi` | any labels, with `type="multi"` | every option above the threshold |

Every answer includes a probability for each option.

## Download

```bash
hf download samratduttaofficial/WaterSheep --local-dir WaterSheep
```

Or with Git (requires [Git LFS](https://git-lfs.com)):

```bash
git clone https://huggingface.co/samratduttaofficial/WaterSheep
```

Then load it from the folder, offline:

```python
ws = pipeline(model="WaterSheep", trust_remote_code=True)
```

## API

Deploy it as a Hugging Face [Inference Endpoint](https://endpoints.huggingface.co), then:

```bash
curl https://YOUR-ENDPOINT -H "Authorization: Bearer $HF_TOKEN" -H "Content-Type: application/json" -d '{"inputs": "I was charged twice.", "parameters": {"question": "Which team should handle this?", "options": ["billing", "shipping", "support"]}}'
```

## JavaScript

No install; runs in the browser:

```html
<script type="module">
  import { decide } from "https://samratduttaofficial.github.io/WaterSheep/watersheep.js";
  console.log(await decide("I was charged twice.", "Which team should handle this?", ["billing", "shipping", "support"]));
</script>
```

With a downloaded copy on your web server, call `load({ base: "WaterSheep/" })` first.

Other languages: run `onnx/model_quantized.onnx` with ONNX Runtime; `watersheep.js` shows the input format.

## Command line

```bash
pip install git+https://github.com/SamratDuttaOfficial/WaterSheep
watersheep --model samratduttaofficial/WaterSheep --question "Which team should handle this?" --options billing,shipping,support --state "I was charged twice."
```

`--serve` runs a local HTTP API on port 8766.

## Train a new model

```bash
git clone https://github.com/SamratDuttaOfficial/WaterSheep
cd WaterSheep
./run.sh
```

Use `run.bat` on Windows.

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
