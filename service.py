from bentoml.frameworks.transformers import TransformersModelArtifact
import bentoml
import torch
import json
from bentoml.adapters import JsonInput, JsonOutput
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@bentoml.env(pip_packages=["transformers", "torch"])
@bentoml.artifacts([TransformersModelArtifact("BartModel")])
class TransformerService(bentoml.BentoService):
    @bentoml.api(input=JsonInput(), batch=False)
    def summary(self, parsed_json):
        with torch.no_grad():
            src_text = parsed_json.get("text")
            model = self.artifacts.BartModel.get("model")
            model.to(device)
            model.eval()
            tokenizer = self.artifacts.BartModel.get("tokenizer")
            with torch.no_grad():
                test_doc = [tokenizer.encode_plus(src_text, add_special_tokens=True,
                                                  max_length=1024,
                                                  pad_to_max_length=True)[
                                "input_ids"]]
                print(test_doc)

                # tensor, gpu
                test_doc = torch.tensor(test_doc)
                test_doc = test_doc.to(device)

                # Generate Summarizaion
                summary_ids = model.generate(test_doc,
                                             num_beams=5,
                                             no_repeat_ngram_size=4,
                                             temperature=1.0, top_k=-1, top_p=-1,
                                             length_penalty=1.0, min_length=1,
                                             max_length=100
                                             ).to(device)

                # Summarization Preprocessing
                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                output = output[:len(output) - output[::-1].find('.')]
                summary = {"summary": output}
                summary = json.dumps(summary)

                return summary


