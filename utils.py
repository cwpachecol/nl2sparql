import torch
# import spacy
from torchtext.data.metrics import bleu_score
import sys

def lreplace(pattern, sub, string):
    """
    Replaces "pattern" in "string" with "sub" if "pattern" starts "string".
    """
    return re.sub("^%s" % pattern, sub, string)

def rreplace(pattern, sub, string):
    """
    Replaces "pattern" in "string" with "sub" if "pattern" ends "string".
    """
    return re.sub("%s$" % pattern, sub, string)



def translate_sentence(model, sentence, question, sparql, device, max_length=50):
    # Load question tokenizer
    tokenize = lambda x: x.split()

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.lower() for token in tokenize(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, question.init_token)
    tokens.append(question.eos_token)

    # Go through each question token and convert to an index
    text_to_indices = [question.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [sparql.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == sparql.vocab.stoi["<eos>"]:
            break

    translated_sentence = [sparql.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]


def bleu(data, model, question, sparql, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, question, sparql, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer, device):
    print("=> Loading checkpoint")

    if device == "cpu":
        checkpoint = torch.load(filename, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
