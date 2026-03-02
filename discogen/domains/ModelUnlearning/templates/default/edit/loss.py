from trainer.unlearn.base import UnlearnTrainer

class CustomUnlearnTrainer(UnlearnTrainer):
    '''
    The compute_loss method below will be called by trainer.train() in the main.py file,
    provided it has EXACTLY the same name and signature as the default compute_loss method from Transformers.Trainer.
    The correct name and signature are already provided: do NOT edit them.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }

        forget_outputs = model(**forget_inputs)
        retain_outputs = model(**retain_inputs)

        # Insert your loss logic here
        loss = ...

        return (loss, forget_outputs) if return_outputs else loss
