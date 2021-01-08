import random
import torch
from torch import optim
import copy

from transformers import GPT2Tokenizer, GPT2LMHeadModel


class GptPatent:
    def __init__(self, model_path='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = '<|endoftext|>'
        self.model = GPT2LMHeadModel.from_pretrained(model_path)

    def predict(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        output = self.model.generate(**inputs,
                                     no_repeat_ngram_size=2,
                                     max_length=100,
                                     num_beams=5,
                                     early_stopping=True)

        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output_text

    def train(self,
              train_data,
              val_data=None,
              batch_size=8,
              num_epoch=5,
              use_cuda=True,
              max_length=512,
              es_patience=2):

        if use_cuda and torch.cuda.is_available():
            print('GPU available, using GPU')
            device = torch.device('cuda')
        else:
            print('GPU not selected or not availale, using CPU')
            device = torch.device('cpu')
            use_cuda = False

        new_init_wb = ['h.0.attn.masked_bias',
                       'h.1.attn.masked_bias',
                       'h.2.attn.masked_bias',
                       'h.3.attn.masked_bias',
                       'h.4.attn.masked_bias',
                       'h.5.attn.masked_bias',
                       'h.6.attn.masked_bias',
                       'h.7.attn.masked_bias',
                       'h.8.attn.masked_bias',
                       'h.9.attn.masked_bias',
                       'h.10.attn.masked_bias',
                       'h.11.attn.masked_bias',
                       'lm_head.weight']
        for name, param in self.model.named_parameters():
            if ('transformer.h' in name) and (name not in new_init_wb):
                param.requires_grad = False

        if use_cuda:
            self.model = self.model.to(device)
 
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        best_val_loss = 10000000
        es_counter = 0
        best_epoch = 0
        for epoch in range(1, num_epoch + 1):
            print(f'Epoch {epoch}')
            random.shuffle(train_data)
            train_chunk = [train_data[x:x+batch_size] for x in range(0, len(train_data), batch_size)]

            if val_data is not None:
                random.shuffle(val_data)
                val_chunk = [val_data[x:x+batch_size] for x in range(0, len(val_data), batch_size)]

            train_loss = 0
            self.model.train()
            print('Training')
            for i, batch in enumerate(train_chunk):
                inputs = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
                if use_cuda:
                    inputs = inputs.to(device)
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                batch_loss = outputs.loss
                batch_loss.backward()
                optimizer.step()

                train_loss += batch_loss.item()

                if i % 10 == 0:
                    print(f'Batch {i + 1}/{len(train_chunk)}: batch_train_loss={batch_loss.item()};')

            if val_data is not None:
                val_loss = 0
                self.model.eval()
                print('Validation')
                for i, batch in enumerate(val_chunk):
                    inputs = self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
                    if use_cuda:
                        inputs = inputs.to(device)
                    with torch.no_grad():
                        outputs = self.model(**inputs, labels=inputs["input_ids"])
                    batch_loss = outputs.loss
                    val_loss += batch_loss.item()
                    if i % 10 == 0:
                        print(f'Batch {i + 1}/{len(val_chunk)}: batch_val_loss={batch_loss.item()};')

            print(f'--------------Epoch {epoch} Completed--------------')
            print(f'epoch_train_loss={train_loss/len(train_chunk)};')
            if val_data is not None:
                print(f'epoch_val_loss={val_loss/len(val_chunk)};')
                if val_loss > best_val_loss:
                    es_counter += 1
                    print(f'epoch_val_loss is not getting better from {best_val_loss}')
                    print(f'es_counter = {es_counter}/{es_patience}')
                else:
                    es_counter = 0
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_model = copy.deepcopy(self.model)

                if es_counter == es_patience:
                    print(f'Early stopping, revert back to best model at epoch {best_epoch}')
                    self.model = best_model
                    break

        if use_cuda:
            self.model = self.model.cpu()

    def save_model(self, save_dir):
        self.model.cpu()
        self.model.save_pretrained(save_dir)
