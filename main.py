from translator.model import ONNXModelExecutor
from whisper_utils import transcriber
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small", help="Model size",
                        choices=["tiny", "base", "small", "medium", "large"])
    args = parser.parse_args()

    model_size = args.model

    sentence = transcriber(model_size)
    inferer = ONNXModelExecutor(src="no", trg="en")
    inferer.load_onnx_model("./translator/No-En-Transformer.onnx")
    output = inferer.infer(sentence)
    print("")
    print("Translated sentence: ") 
    print(output)