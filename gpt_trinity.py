from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import Adam 
import torch
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef # CoLA metrics

from utils import *
from models import *
import argparse

import os
import datetime
import time


def train(model, train_dataset, valid_dataset, args):
    model.to(args.device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=args.lr)

    set_seed(args.seed)
    itr = 1
    p_itr = 100

    total_loss = 0
    total_len = 0
    total_correct = 0

    pred_labels = []
    gold_labels = []

    best_acc = 0
    if args.verbose:
        print(f"length of trainset = {len(train_dataset)}")
    for epoch in range(args.epoch):
        """
            Train
        """
        model.train()
        t0 = time.time()
        for x_train, y_train in train_loader:
            optimizer.zero_grad()
            x_train['input_ids'] = x_train['input_ids'].to(args.device)
            x_train['attention_mask'] = x_train['attention_mask'].to(args.device)
            if 'token_type_ids' in x_train: # TODO : WiC Task에서 작동하는지 확인
                x_train['token_type_ids'] = x_train['token_type_ids'].to(args.device)
            y_train = y_train.to(args.device)
            out = model(input_ids=x_train['input_ids'], attention_mask=x_train['attention_mask'], token_type_ids=x_train['token_type_ids'], labels=y_train)
            loss, logits = out.loss, out.logits
            pred = torch.argmax(F.softmax(logits, dim=1), dim=1)

            pred_labels += pred.tolist()
            gold_labels += y_train.tolist()
            correct = pred.eq(y_train)

            total_loss += loss.item()
            total_len += len(y_train)
            total_correct += correct.sum().item()


            loss.backward()
            optimizer.step()

            if args.verbose and itr % p_itr == 0:
                print(f"[Epoch {epoch+1}/{args.epoch}] Iteration {itr} -> loss {total_loss/total_len:.4f}, mattew {matthews_corrcoef(gold_labels, pred_labels):.4f}, acc {(total_correct/total_len)*100:.4f}, time {time.time()-t0:.2f}")
                t0 = time.time()
                     
            itr += 1
        
        print(f"[Epoch {epoch+1}] loss {total_loss/total_len}, acc {(total_correct/total_len)*100}")
        
        """
            Validation
        """
        model.eval()

        val_total_loss = 0
        val_total_len = 0
        val_total_correct = 0
        val_pred_labels = []
        val_gold_labels = []

        for x_valid, y_valid in valid_loader:
            optimizer.zero_grad()
            x_valid['input_ids'] = x_valid['input_ids'].to(args.device)
            x_valid['attention_mask'] = x_valid['attention_mask'].to(args.device)
            if 'token_type_ids' in x_valid: # TODO : WiC Task에서 작동하는지 확인
                x_valid['token_type_ids'] = x_valid['attention_mask'].to(args.device)
            y_valid = y_valid.to(args.device)
            out = model(input_ids=x_valid['input_ids'], attention_mask=x_valid['attention_mask'],labels=y_valid)
            loss, logits = out.loss, out.logits
            pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
            correct = pred.eq(y_valid)
            
            val_pred_labels += pred.tolist()
            val_gold_labels += y_valid.tolist()
            val_total_loss += loss.item()
            val_total_len += len(y_valid)
            val_total_correct += correct.sum().item()
        
        valid_acc = (val_total_correct/val_total_len)*100

        print(f"validation {epoch+1} : loss {val_total_loss/val_total_len:.4f}, mattew {matthews_corrcoef(val_gold_labels, val_pred_labels):.4f}, acc {valid_acc:.2f}")
        
        if valid_acc > best_acc:
            curr_time = datetime.datetime.today().strftime("%m-%d-%H%M")
            if not os.path.exists('./result'):
                os.mkdir('./result')
            best_acc = valid_acc
            torch.save(model.state_dict(), os.path.join('./result', args.task + "-" +args.model+ "-" + curr_time + f"({best_acc:.2f})"))
            

        total_loss = 0
        total_len = 0
        total_correct = 0
        pred_labels = []
        gold_labels = []


def eval(model, device, valid_dataset, args):
    """
        Evaluate saved models
    """
    model.load_state_dict(torch.load(os.path.join(args.model_dir, args.saved_state)))
    model.to(device)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=args.lr)

    model.eval()

    val_total_loss = 0
    val_total_len = 0
    val_total_correct = 0
    val_pred_labels = []
    val_gold_labels = []

    for x_valid, y_valid in valid_loader:
        optimizer.zero_grad()
        x_valid['input_ids'] = x_valid['input_ids'].to(device)
        x_valid['attention_mask'] = x_valid['attention_mask'].to(device)
        y_valid = y_valid.to(device)
        out = model(input_ids=x_valid['input_ids'], attention_mask=x_valid['attention_mask'], labels=y_valid)
        loss, logits = out.loss, out.logits
        pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
        correct = pred.eq(y_valid)
        
        val_pred_labels += pred.tolist()
        val_gold_labels += y_valid.tolist()
        val_total_loss += loss.item()
        val_total_len += len(y_valid)
        val_total_correct += correct.sum().item()
    
    valid_acc = (val_total_correct/val_total_len)*100

    print(f"Evaluation loss {val_total_loss/val_total_len:.4f}, mattew {matthews_corrcoef(val_gold_labels, val_pred_labels):.4f}, acc {valid_acc:.2f}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True, help='Task Choice. CoLA|WiC|BoolQ|COPA')
    parser.add_argument("--mode", type=str, required=True, help='Execution mode. train|dev')
    
    parser.add_argument("--data_dir", type=str, default='./corpus', help='Data directory, default=\'./corpus\'')
    parser.add_argument("--model_dir", type=str, default='./result', help='Model save directory, default=\'./result\'')
    parser.add_argument("--saved_state", type=str, required=False, help='Saved model')
    parser.add_argument("--device", type=str, default='cuda', help='Device information.')
    parser.add_argument("--verbose", type=bool, default=True, help='Verbose option.')
    parser.add_argument("--pretrained_model", type=str, default="skt/ko-gpt-trinity-1.2B-v0.5")

    parser.add_argument("--model", type=str, default='vanila_GPT', help='Model choice.')
    parser.add_argument("--seed", type=int, default=42, help='Manual seed.')
    parser.add_argument("--epoch", type=int, default=3, help='Train epoch.')
    parser.add_argument("--batch_size", type=int, default=2, help='Batch size.')
    parser.add_argument("--lr", type=float, default=1e-3, help='Learning rate.')
    args = parser.parse_args()
    args.device = torch.device(args.device)


    if args.model == "vanila_GPT":
        model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=2)

    train_file, valid_file, test_file = file_selector(args)
    if args.mode == "train":
        train_dataset = dataset_selector(train_file, args)
        valid_dataset = dataset_selector(valid_file, args)
        train(model, train_dataset, valid_dataset, args)
    if args.mode == "dev":
        valid_dataset = dataset_selector(valid_file, args)
        eval(model, args.device, valid_dataset, args)



if __name__ == "__main__":
    main()
