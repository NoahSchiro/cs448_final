import sys

import torch

# This must be the same as the model training
CONTEXT  = 50

model     = torch.load("./results/transformer_best.pth")
tokenizer = torch.load("./results/tokenizer.pth")
vocab     = torch.load("./results/vocab.pth")
 
def text_preprocessing(text):

    tokenized = tokenizer(text)

    if len(tokenized) < CONTEXT:
        # Append the "unkown" token till we get to CONTEXT 
        while len(tokenized) < CONTEXT:
            tokenized.append("<unk>")

    # Or slice the list down
    else:
        tokenized = tokenized[:CONTEXT]

    # 1.Turn tokens into ints
    # 2.Turn list of ints into tensor
    # 3.Turn it into a column tensor
    return torch.tensor(vocab(tokenized))

def classify_sentiment(model, input):
    input = text_preprocessing(input)
    probabilities = model(input)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class, probabilities

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference_transformer.py <text>")
        sys.exit(1)

    text = sys.argv[1]

    predicted_class, probabilities = classify_sentiment(model, text)

    print(f"Text: {text}")
    print(f"Predicted Sentiment: {'Positive' if predicted_class == 0 else 'Negative'}")
    print(f"Probability (Positive): {probabilities[0, 1]:.4f}")
    print(f"Probability (Negative): {probabilities[0, 0]:.4f}")
