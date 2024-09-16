import os
import argparse
from argparse import Namespace
from pprint import pprint
from functools import partial

import numpy # for gradio hot reload
import gradio as gr

import torch

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList)

from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
import pandas as pd
from tqdm import tqdm
import json

def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    """Command line argument specification"""

    parser = argparse.ArgumentParser(description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API")

    parser.add_argument(
        "--run_gradio",
        type=str2bool,
        default=False,
        help="Whether to launch as a gradio demo. Set to False if not installed and want to just run the stdout version.",
    )
    parser.add_argument(
        "--demo_public",
        type=str2bool,
        default=False,
        help="Whether to expose the gradio demo to the internet.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=123,
        help="Seed for setting the torch global rng prior to generation.",
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=True,
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7,
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,
        help="Number of beams to use for beam search. 1 is normal greedy decoding",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=False,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="The amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_bigrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--select_green_tokens",
        type=str2bool,
        default=True,
        help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
    )
    parser.add_argument(
        "--skip_model_load",
        type=str2bool,
        default=False,
        help="Skip the model loading to debug the interface.",
    )
    parser.add_argument(
        "--seed_separately",
        type=str2bool,
        default=True,
        help="Whether to call the torch seed function before both the unwatermarked and watermarked generate calls.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=False,
        help="Whether to run model in float16 precsion.",
    )
    args = parser.parse_args()
    return args
    
def generate(prompt, args, model=None, device=None, tokenizer=None, watermark=False, system_prompt=""):
    """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
       and generate watermarked text by passing it to the generate method of the model
       as a logits processor. """
    
    print(f"Generating with {args}")

    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=args.gamma,
                                                    delta=args.delta,
                                                    seeding_scheme=args.seeding_scheme,
                                                    select_green_tokens=args.select_green_tokens)

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)

    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True, 
            top_k=0,
            temperature=args.sampling_temp
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams
        ))

    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )
    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]), 
        **gen_kwargs
    )
    if args.prompt_max_length:
        pass
    elif hasattr(model.config,"max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings-args.max_new_tokens
    else:
        args.prompt_max_length = 2048-args.max_new_tokens

    full_prompt = f"{system_prompt}\n\n{prompt}"
    
    tokd_input = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_max_length).to(device)
    truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

    torch.manual_seed(args.generation_seed)

    if not watermark:
        output = generate_without_watermark(**tokd_input)
    else:
        output = generate_with_watermark(**tokd_input)


    # need to isolate the newly generated tokens
    output = output[:,tokd_input["input_ids"].shape[-1]:]

    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    return (redecoded_input,
            int(truncation_warning),
            decoded_output,
            args) 
            # decoded_output_with_watermark)

#def detect(input_text, args, device=None, tokenizer=None):
#    """Instantiate the WatermarkDetection object and call detect on
#        the input text returning the scores and outcome of the test"""
#    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
#                                        gamma=args.gamma,
#                                        seeding_scheme=args.seeding_scheme,
#                                        device=device,
#                                        tokenizer=tokenizer,
#                                        z_threshold=args.detection_z_threshold,
#                                        normalizers=args.normalizers,
#                                        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
#                                        select_green_tokens=args.select_green_tokens)
#    if len(input_text)-1 > watermark_detector.min_prefix_len:
#        score_dict = watermark_detector.detect(input_text)
#        # output = str_format_scores(score_dict, watermark_detector.z_threshold)
#        output = list_format_scores(score_dict, watermark_detector.z_threshold)
#    else:
#        # output = (f"Error: string not long enough to compute watermark presence.")
#        output = [["Error","string too short to compute metrics"]]
#        output += [["",""] for _ in range(6)]
#    return output, args

def detect(input_text, args, device=None, tokenizer=None, window_size=None, window_stride=None):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""

    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=args.gamma, # should match original setting
                                        seeding_scheme=args.seeding_scheme, # should match original setting
                                        device=device, # must match the original rng device type
                                        tokenizer=tokenizer,
                                        z_threshold=args.detection_z_threshold,
                                        normalizers=args.normalizers,
                                        ignore_repeated_ngrams=args.ignore_repeated_bigrams,
                                        select_green_tokens=args.select_green_tokens)
    score_dict = watermark_detector.detect(input_text, window_size=window_size, window_stride=window_stride)
    # output = str_format_scores(score_dict, watermark_detector.z_threshold)
    output = list_format_scores(score_dict, watermark_detector.z_threshold)
    return output, args

def load_model(args):
    """Load and return the model and tokenizer"""

    args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5","T0"]])
    args.is_decoder_only_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt","opt","bloom"]])
    # if args.is_seq2seq_model:
    #     model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    # elif args.is_decoder_only_model:
    #     if args.load_fp16:
    #         model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.float16, device_map='auto')
    #     else:
    #         model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    # else:
    #     raise ValueError(f"Unknown model type: {args.model_name_or_path}")
    import os
    model = model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, token=os.getenv("HUG_TOKEN"))
    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.load_fp16: 
            pass
        else: 
            model = model.to(device)
    else:
        device = "cpu"
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=os.getenv("HUG_TOKEN"))

    return model, tokenizer, device

def format_names(s):
    """Format names for the gradio demo interface"""
    s=s.replace("num_tokens_scored","Tokens Counted (T)")
    s=s.replace("num_green_tokens","# Tokens in Greenlist")
    s=s.replace("green_fraction","Fraction of T in Greenlist")
    s=s.replace("z_score","z-score")
    s=s.replace("p_value","p value")
    s=s.replace("prediction","Prediction")
    s=s.replace("confidence","Confidence")
    return s
    
def list_format_scores(score_dict, detection_threshold):
    """Format the detection metrics into a gradio dataframe input format"""
    lst_2d = []
    # lst_2d.append(["z-score threshold", f"{detection_threshold}"])
    for k,v in score_dict.items():
        if k=='green_fraction': 
            lst_2d.append([format_names(k), f"{v:.1%}"])
        elif k=='confidence': 
            lst_2d.append([format_names(k), f"{v:.3%}"])
        elif isinstance(v, float): 
            lst_2d.append([format_names(k), f"{v:.3g}"])
        elif isinstance(v, bool):
            lst_2d.append([format_names(k), ("Watermarked" if v else "Human/Unwatermarked")])
        else: 
            lst_2d.append([format_names(k), f"{v}"])
    if "confidence" in score_dict:
        lst_2d.insert(-2,["z-score Threshold", f"{detection_threshold}"])
    else:
        lst_2d.insert(-1,["z-score Threshold", f"{detection_threshold}"])
    return lst_2d

def change_to_dict_w_keys(ll, suffix):
    """"""
    return {
        f"{k[0]} ({suffix})":k[1] for k in ll
    }
    
if __name__ == "__main__":

    prompts_df = pd.read_csv("data/reddit_questions.csv",sep=";")
    args = parse_args()
    model, tokenizer, device = load_model(args)

    RES_FILE = open("results.json", "a")
    
    for idx, row in tqdm(prompts_df.iterrows()):
        prompt = row.text    
        SYS1 = "You are a redditor. Answer the questions with a university graduate level english. Don't give short answers. At least 4-5 sentences and 100 + words."
        
        SYS2 = "Paraphrase the following;"
        
        prompt = "How do you remove black rings under your eye from pulling all nighters?"
        
        inp, trunc, out, _ = generate(prompt, args, model, device, tokenizer, True, SYS1)
        

        result_wm , _ = detect(out, args, device, tokenizer, "25,50", 1)
        result_wm = change_to_dict_w_keys(result_wm, "WM")

        inp2, trunc2, out2, _ = generate(out, args, model, device, tokenizer, False, SYS2)

        result_p , _ = detect(out2, args, device, tokenizer, "25,50", 1)
        result_p = change_to_dict_w_keys(result_p, "Para")

        final = {"id":idx, **result_wm, **result_p, "wm_text":out, "p_text": out2}

        with open("results.json", "a") as RES_FILE:
            RES_FILE.write(json.dumps(final, default=str) + ",")


