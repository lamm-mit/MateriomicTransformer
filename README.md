# MateriomicTransformer

### Generative Pretrained Autoregressive Transformer Graph Neural Network applied to the Analysis and Discovery of Materials

MateriomicTransformer is a flexible language-model based deep learning strategy, applied here to solve complex forward and inverse problems in protein modeling, based on an attention neural network that integrates transformer and graph convolutional architectures in a causal multi-headed graph mechanism, to realize a generative pretrained model. The model is applied to predict secondary structure content (per-residue level and overall content), protein solubility, and sequencing tasks. Further trained on inverse tasks, the model is rendered capable of designing proteins with these properties as target features. The model is formulated as a general framework, completely prompt-based, and can be adapted for a variety of downstream tasks. We find that adding additional tasks yields emergent synergies that the model exploits in improving overall performance, beyond what would be possible by training a model on each dataset alone. Case studies are presented to validate the method, yielding protein designs specifically focused on structural proteins, but also exploring the applicability in the design of soluble, antimicrobial biomaterials. While our model is trained to ultimately perform 8 distinct tasks, with available datasets it can be extended to solve additional problems. In a broader sense, this work illustrates a form of multiscale modeling that relates a set of ultimate building blocks (here, byte-level utf8 characters that define the nature of the physical system at hand) to complex output. This materiomic scheme captures complex emergent relationships between universal building block and resulting properties via a synergizing learning capacity to express a set of potentialities embedded in the knowledge used in training, via the interplay of universality and diversity.

### Significance
Predicting the properties of materials based on a flexible description of their structure, environment or process is a long-standing challenge in multiscale modeling. Our MaterioFormer language-model, trained to solve forward and inverse tasks, incorporates a deep learning capacity through attention and graph strategies, to yield a multimodal approach to model and design materials. Since our model is prompt-based and information is encoded consistently via byte-level utf8 tokenization, it can process diverse modalities of information, such as sequence data, description of tasks, and numbers and offers a flexible workflow that integrates human intelligence and AI. Autoregressive training, using pre-training against a large unlabeled dataset, allows for straightforward  adjustment of specific objectives. 

### Installation and use

Install OmegaFold and DSSP

```
pip install git+https://github.com/HeliXonProtein/OmegaFold.git
sudo apt-get install dssp
```
Install MateriomicTransformer
```
git clone https://github.com/lamm-mit/MateriomicTransformer/
cd MateriomicTransformer
pip install -e .
```

Then open the Jupyter notebook for training/inference. 

```
from   MateriomicTransformer import MoleculeTransformerGPT, count_parameters

mask_token=95 #token to be used for masking 
num_words=256 #number of words
max_length=64 #max length, just for inference (number of generated tokens)

model = MoleculeTransformerGPT(
        dim=256,
        depth=12,
        logits_dim=num_words, #number of tokens 
        max_tokens = num_words,
        dim_head = 32,
        heads = 8,
        dropout = 0.1,
        ff_mult = 4,
        one_kv_head=False,
        embed_dim = 32, #for input sequence
        concat_pos_encoding = False, #if True, then pos encoding will be added to the input; and pos_fourier_graph_dim=embed_dim
        pos_fourier_graph_dim  = 32,  #only used if concat_pos_encoding == True
        use_null_kv = False, #True, #False,
        FF_conv_kernel= 0, #if >0 use resnet bloack in FF layer 
        FF_inner_conv_kernel=0,
        FF_glu=False,
        GNN_layers=3,
        GNN_att_threshold_min=0., #below is set to 0, if>0
        GNN_att_threshold_max=1,  #above is set to this value, if <1
        GNN_have_skip = True, #if GNN has skip connections
        GNN_add_identity= False, #whether to add identity to adjancy or feature matrix
        GNN_clamp_att_after_identity = True, #whether to clamp attention after identity is added (only used IF identity is added)    
).to(device)

count_parameters (model)
lr=0.0002
optimizer = optim.Adam(model.parameters() , lr=lr)

x = torch.randint(0, num_words, (2, max_length)).to(device)
print (x.shape)

loss = model(output=x, 
        return_loss=True, 
        mask_prob=0.15, #mask out random elements in the sequence 
        mask_token= mask_token, #replace random token with mask_token, with mask_token_prob
        mask_token_prob=0.15,
        task_end_token=task_end_token, #only consider tokens after the first occurrence of task_end_token in loss calculation
        ignore_padding_zeros = False, #if True, ignore 0 token. If task_end_token is defined, tokens before the first occurence of task_end_token will be set to 0, and also ignored.
        )
loss.backward()

start_tokens = torch.randint(0, num_words, (2, 20)).to(device)

result=model.generate(output=start_tokens,
        tokens_to_generate=max_length,
        temperature = 1.,
        filter_thres = 0.9,
    )
print (result.shape)
```

Weights of trained model: Download [here...](https://www.dropbox.com/scl/fi/timpki8r2pvgoc4rw1nl7/statedict_V4031.pt?rlkey=ixndtsc6mndcw9rakd38ge771&dl=0)

```
fname='statedict_V4031.pt'
model.load_state_dict(torch.load(fname))
```
