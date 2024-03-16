# Copyright (c) OpenMMLab. All rights reserved.
import torch
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
# from xtuner.dataset.map_fns import openorca_map_fn, template_map_fn_factory
from xtuner.dataset.map_fns import template_map_fn_factory
from xtuner.engine.hooks import (DatasetInfoHook, EvaluateChatHook,
                                 VarlenAttnArgsToMessageHubHook)
from xtuner.engine.runner import TrainLoop
from xtuner.model import SupervisedFinetune
from xtuner.utils import PROMPT_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = 'meta-llama/Llama-2-7b-chat-hf'  #改路径 ！！
use_varlen_attn = False

# Data
data_path = '/LlamaIndex-ScienceChat/code/classifier/xtuner_data_train6.json' ### 改路径！！
prompt_template = PROMPT_TEMPLATE.llama2_chat_ours ###这个要额外改一下源码，他的不对！！！！！
max_length = 4090 #这个不确定，每条不定长，看情况，先拉满
pack_to_max_length = True

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 16
dataloader_num_workers = 0
max_epochs = 1# 可以改一下epoch
optim_type = AdamW
lr = 2e-4
betas = (0.9, 0.95)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.025

# Save
save_steps = 100
save_total_limit = 1  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 100

SYSTEM = "You are an AI assistant that helps users answer multiple-choice questions based on the provided context and options. The context are to support your decision-making process. The options are the possible answers to the question. Select only one option with only one letter from (A, B, C, D, or E) as your answer."
evaluation_inputs = [
    "Below is an instruction that describes a task, paired with an input that provides further context that helps you select one of the most correct option from the given options for the given question. Write a response that appropriately completes the request.\n\n### Instruction:\nCarefully analyze the question and options below. Choose the most appropriate option based on the provided context. Respond with only the letter (A, B, C, D, or E) corresponding to your answer choice.\n\n### Input:\nContext: Potato virus Y | RT-PCR | Reverse transcriptase polymerase chain reaction (RT-PCR) has become a powerful and effective method for detection of potato plant viruses within potato plant material and even dormant potatoes. Only a minute piece of plant material is required for analysis using RT-PCR. Considering the protocol described within this thesis, 0.1 g of plant material is enough for 14 500 separate reactions. During a RT-PCR specific target RNA sequences are amplified exponentially into DNA copies.\n\nFor this to occur, however, the RNA of the virus must first be transcribed to DNA by means of a reverse transcriptase polymerase. This polymerase synthesizes a DNA strand using the RNA as template. This results in a DNA/RNA complex.\n\nFor synthesis of a DNA strand from the RNA template only the reverse primer is required since the RNA is a single strand arranged from 5\u2019 to 3\u2019. Subsequently, the newly synthesized DNA strand is used as a template for traditional PCR.\n\n||\n\nAlphaguttavirus | Life cycle | DNA-templated transcription is the method of transcription. Sulfolobus newzealandicus serve as the natural host.\n\nQuestion: What is the method of transcription in the life cycle of viruses in the genus Becurtovirus?\n\nOptions:\nA: RNA-templated transcription is the method of transcription.\n\nB: Transcription occurs through a unique mechanism specific to viruses in the genus Becurtovirus.\n\nC: Reverse transcription is the method of transcription.\n\nD: DNA-templated transcription is the method of transcription.\n\nE: Transcription does not occur in the life cycle of viruses in the genus Becurtovirus.\n\n\nAnswer:",
    "Below is an instruction that describes a task, paired with an input that provides further context that helps you select one of the most correct option from the given options for the given question. Write a response that appropriately completes the request.\n\n### Instruction:\nCarefully analyze the question and options below. Choose the most appropriate option based on the provided context. Respond with only the letter (A, B, C, D, or E) corresponding to your answer choice.\n\n### Input:\nContext: Chikungunya fever | Viral replication | The virus consists of four nonstructural proteins and three structural proteins. The structural proteins are the capsid and two envelope glycoproteins: E1 and E2, which form heterodimeric spikes on the viron surface. E2 binds to cellular receptors in order to enter the host cell through receptor-mediated endocytosis. E1 contains a fusion peptide which, when exposed to the acidity of the endosome in eukaryotic cells, dissociates from E2 and initiates membrane fusion that allows the release of nucleocapsids into the host cytoplasm, promoting infection. The mature virion contains 240 heterodimeric spikes of E2/E1, which after release, bud on the surface of the infected cell, where they are released by exocytosis to infect other cells.\n\n||\n\nCorticovirus | Genome | The genome is not segmented, constitutes 13% of the virus's weight and contains a single molecule of circular, supercoiled, double-stranded DNA of 10 kilobases in length. The genome has a g + c content of 43%. It encodes ~21 proteins. Transcription is organised into three operons. Replication of the genome is via a rolling-circle mechanism, initiated by the virus encoded endonuclease P12.\n\nQuestion: What is the role of the viral fiber glycoproteins in the life cycle of Ichtadenovirus?\n\nOptions:\nA: The viral fiber glycoproteins are involved in the replication of the viral DNA.\n\nB: The viral fiber glycoproteins code for 40 proteins in the genome.\n\nC: The viral fiber glycoproteins are responsible for the icosahedral geometry of the virus.\n\nD: The viral fiber glycoproteins mediate endocytosis of the virus into the host cell.\n\nE: The viral fiber glycoproteins are responsible for the lysis of the host cell.\n\n\nAnswer:",
     "Below is an instruction that describes a task, paired with an input that provides further context that helps you select one of the most correct option from the given options for the given question. Write a response that appropriately completes the request.\n\n### Instruction:\nCarefully analyze the question and options below. Choose the most appropriate option based on the provided context. Respond with only the letter (A, B, C, D, or E) corresponding to your answer choice.\n\n### Input:\nContext: 3 Geminorum | Summary | The brighter component is the variable blue supergiant. The companion is 2.5 magnitudes fainter. The separation is about 0.6 arc-seconds.\n\nThere is also a much fainter, approximately 14th magnitude, star 14\" away.Faint H\u03b1 emission lines have been detected in the spectrum of 3 Geminorum, but this is not usually expressed in published spectral classifications. An \"e\" is only occasionally appended to the spectral type to reflect the emission lines. 3 Geminorum has frequently been classified as a normal supergiant (luminosity class Ib), although a bright supergiant (Ia) luminosity class is now preferred.3 Geminorum can be occulted by the Moon.\n\nObservations of these occultations can give information about the angular diameter of a star, or about close companions. Occultations of 3 Geminorum have been observed, but no double or diameter information has been published. == References ==\n\n||\n\nV392 Persei | Dwarf nova | A U Geminorum-type variable star or dwarf nova is a type of cataclysmic variable star consisting of a close binary star system in which one of the components is a white dwarf that accretes matter from a cool main sequence or subgiant companion. V392 Persei was discovered in 1970 and received its variable star designation a year later. It is normally visual magnitude 17.4 and experiences outbursts of 2-3 magnitudes.\n\nIts spectrum in the quiescent state has been studied and only the cool star is detected. The spectrum shows emission lines of hydrogen-alpha (H\u03b1) and both neutral and ionised helium. The brightest recorded observations is at magnitude 5.6.\n\nQuestion: What is the significance of the faint H\u03b1 emission lines detected in the spectrum of 3 Geminorum?\n\nOptions:\nA: The emission lines indicate that 3 Geminorum is a bright supergiant (luminosity class Ia).\n\nB: The emission lines indicate that 3 Geminorum is a normal supergiant (luminosity class Ib).\n\nC: The emission lines indicate that 3 Geminorum is a small amplitude pulsating variable.\n\nD: The emission lines indicate that 3 Geminorum is a blue supergiant star.\n\nE: The emission lines indicate that 3 Geminorum is a close double star.\n\n\nAnswer:"
]
# 答案： DDA

#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')

model = dict(
    type=SupervisedFinetune,
    use_varlen_attn=use_varlen_attn,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')),
    lora=dict(
        type=LoraConfig,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'))

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=process_hf_dataset,
    dataset=dict(
       type=load_dataset, path='json', data_files=dict(train=data_path)),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=None,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=tokenizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        system=SYSTEM,
        prompt_template=prompt_template)
]

if use_varlen_attn:
    custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)