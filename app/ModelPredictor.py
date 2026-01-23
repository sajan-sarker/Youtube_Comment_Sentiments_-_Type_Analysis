import torch
import torch.nn.functional as F

class ModelPredictor:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def get_predict(self, comments, batch_size=8):
        # if a single comment passed, then wrap it in a list
        single = False
        if isinstance(comments, str):
            comments = [comments]
            single = True

        sentiments = []
        types = []
        senti_proba = []
        type_proba = []

        self.model.eval()
        for i in range(0, len(comments), batch_size):
            batch_comment = comments[i:i + batch_size]

            comment = self.tokenizer(batch_comment, padding=True, return_attention_mask=True, truncation=True, max_length=128, return_tensors='pt')
            ids, attention_mask = comment['input_ids'].to(self.device), comment['attention_mask'].to(self.device)

            with torch.no_grad():
                # use the tensor variable, not the original list
                y_sen, y_ty = self.model(ids, attention_mask)

            # Compute probabilities
            prob_sen = F.softmax(y_sen, dim=1)
            prob_ty = F.softmax(y_ty, dim=1)

            # Get max probability for each sample in the batch
            prob_sen = torch.max(prob_sen, dim=1).values.cpu().tolist()
            prob_ty = torch.max(prob_ty, dim=1).values.cpu().tolist()

            # Round probabilities to 2 decimals
            prob_sen = [round(p, 4) for p in prob_sen]
            prob_ty = [round(p, 4) for p in prob_ty]

            # Add to main lists
            senti_proba.extend(prob_sen)
            type_proba.extend(prob_ty)

            sentiments.extend(torch.argmax(y_sen, dim=1).cpu().tolist())
            types.extend(torch.argmax(y_ty, dim=1).cpu().tolist())

        if single:
            return sentiments[0], types[0], senti_proba[0], type_proba[0]
        return sentiments, types, senti_proba, type_proba