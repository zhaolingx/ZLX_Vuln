import sys
# sys.path.append('/home/EPVD/')
import parserTool.parse as ps
from c_cfg import C_CFG
from parserTool.utils import *
from parserTool.parse import Lang
import json
import pickle
import logging
import numpy as np
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}
# config_class, model_class, tokenizer_class = MODEL_CLASSES["roberta"]
# tokenizer = tokenizer_class.from_pretrained("microsoft/codebert-base", do_lower_case=True)

logger = logging.getLogger(__name__)

def extract_pathtoken(source, path_sequence):
    seqtoken_out = []
    for path in path_sequence:
        seq_code = ''
        for line in path:
            if (line in source):
                seq_code += source[line]
        seqtoken_out.append(seq_code)
        if len(seqtoken_out) > 10:
            break
    if len(path_sequence) == 0:
        seq_code = ''
        for i in source:
            seq_code += source[i]
        seqtoken_out.append(seq_code)
    seqtoken_out = sorted(seqtoken_out, key=lambda i: len(i), reverse=False)
    return seqtoken_out
    
def main():
    output = open('short_4path_javadata_nobalance.pkl', 'wb')
    path_dict = {}
    state_dict = {}
    num_id = 0
    sum_ratio = 0
    num_path_dict = {}
    with open("../dataset/cdata/nobalance/train_cdata.jsonl", encoding="utf8") as f:
        for line in f:
            num_id += 1
            if num_id%100 == 0:
                print(num_id, flush=True)
            
            js = json.loads(line.strip())
            
            clean_code, code_dict = remove_comments_and_docstrings(js['func'], 'java')
            g = C_CFG()
            code_ast = ps.tree_sitter_ast(clean_code, Lang.C)
            code_ast_java = ps.tree_sitter_ast(clean_code, Lang.JAVA)
            s_ast = g.parse_ast_file(code_ast_java.root_node)
            tokens_index = tree_to_token_index(code_ast_java.root_node)
            code = clean_code.split('\n')
            code_tokens = [index_to_code_token(x, code) for x in tokens_index]
            index_to_code = {}
            for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
                index_to_code[index] = (idx, code)
            DFG, _ = g.DFG_java(code_ast_java.root_node, index_to_code, {})
            num_path, cfg_allpath, _, ratio = g.get_allpath()
            path_tokens1 = extract_pathtoken(code_dict, cfg_allpath)
            sum_ratio += ratio
            path_dict[js['idx']] = path_tokens1, cfg_allpath
            #print("num_paths:", num_path)
    print("train file finish...", flush=True)

    with open("../dataset/cdata/nobalance/valid_cdata.jsonl", encoding="utf-8") as f:
        for line in f:
            num_id += 1
            if num_id%100==0:
                print(num_id, flush=True)
            js = json.loads(line.strip())
            clean_code, code_dict = remove_comments_and_docstrings(js['func'], 'java')
            g = C_CFG()
            code_ast = ps.tree_sitter_ast(clean_code, Lang.C)
            code_ast_java = ps.tree_sitter_ast(clean_code, Lang.JAVA)
            s_ast = g.parse_ast_file(code_ast_java.root_node)
            tokens_index = tree_to_token_index(code_ast_java.root_node)
            code = clean_code.split('\n')
            code_tokens = [index_to_code_token(x, code) for x in tokens_index]
            index_to_code = {}
            for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
                index_to_code[index] = (idx, code)
            DFG, _ = g.DFG_java(code_ast_java.root_node, index_to_code, {})
            num_path, cfg_allpath, _, ratio = g.get_allpath()
            path_tokens1 = extract_pathtoken(code_dict, cfg_allpath)
            sum_ratio += ratio
            path_dict[js['idx']] = path_tokens1, cfg_allpath
            #print("num_paths:", num_path)
    print("valid file finish...", flush=True)

    with open("../dataset/cdata/nobalance/test_cdata.jsonl", encoding="utf-8") as f:
        for line in f:
            num_id += 1
            if num_id%100==0:
                print(num_id, flush=True)
            js = json.loads(line.strip())
            clean_code, code_dict = remove_comments_and_docstrings(js['func'], 'java')
            g = C_CFG()
            code_ast = ps.tree_sitter_ast(clean_code, Lang.C)
            code_ast_java = ps.tree_sitter_ast(clean_code, Lang.JAVA)
            s_ast = g.parse_ast_file(code_ast_java.root_node)
            tokens_index = tree_to_token_index(code_ast_java.root_node)
            code = clean_code.split('\n')
            code_tokens = [index_to_code_token(x, code) for x in tokens_index]
            index_to_code = {}
            for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
                index_to_code[index] = (idx, code)
            DFG, _ = g.DFG_java(code_ast_java.root_node, index_to_code, {})
            num_path, cfg_allpath, _, ratio = g.get_allpath()
            sum_ratio += ratio
            path_tokens1 = extract_pathtoken(code_dict, cfg_allpath)
            path_dict[js['idx']] = path_tokens1, cfg_allpath
            #print("num_paths:", num_path)
    print("test file finish...", flush=True)
    print(sum_ratio/num_id, flush=True)
    # Pickle dictionary using protocol 0.
    pickle.dump(path_dict, output)
    output.close()

if __name__=="__main__":
    main()
