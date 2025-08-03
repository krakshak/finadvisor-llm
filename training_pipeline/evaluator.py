import math
import evaluate

class EvaluatorModule:
    def __init__(self):
        self.perplexity_metric = evaluate.load("perplexity")

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        # Compute loss and perplexity
        loss = logits["eval_loss"] if isinstance(logits, dict) and "eval_loss" in logits else logits
        perplexity = math.exp(loss) if loss < 100 else float("inf")
        return {"perplexity": perplexity}
