## 🔧 Requirements

Use the following code to install requirements. If you have any problems, using the models, please follow the instructions below.

```
pip install -r requirements.txt
```

### Closed source models
To use closed source models, you should save your API keys in a file named `.env` like the example below .<br />
GEMINI_API_KEY=_YOUR_KEY_<br />
AZURE_OPENAI_API_KEY=_YOUR_KEY_<br />
AZURE_OPENAI_ENDPOINT=_YOUR_KEY_<br />
ANTHROPIC_API_KEY=_YOUR_KEY_<br />

### Open source models
Unfortunately, different MLLMs need diffenrent versions of `transformers` package and we could not find a version that support all of the models. Please use the following versions for each MLLM.<br />
* `LLaVA-Med`: Use `transformers` 4.36.2
* `RadFM`: Use `transformers` 4.28.1
* `MedFlamingo`: Use `transformers` 4.44.2 and install `open-flamingo` package
* `Other MLLMs`: Use `transformers` 4.44.2
