# MateriomicTransformer

### Generative Pretrained Autoregressive Transformer Graph Neural Network applied to the Analysis and Discovery of Materials

MateriomicTransformer is a flexible language-model based deep learning strategy, applied here to solve complex forward and inverse problems in protein modeling, based on an attention neural network that integrates transformer and graph convolutional architectures in a causal multi-headed graph mechanism, to realize a generative pretrained model. The model is applied to predict secondary structure content (per-residue level and overall content), protein solubility, and sequencing tasks. Further trained on inverse tasks, the model is rendered capable of designing proteins with these properties as target features. The model is formulated as a general framework, completely prompt-based, and can be adapted for a variety of downstream tasks. We find that adding additional tasks yields emergent synergies that the model exploits in improving overall performance, beyond what would be possible by training a model on each dataset alone. Case studies are presented to validate the method, yielding protein designs specifically focused on structural proteins, but also exploring the applicability in the design of soluble, antimicrobial biomaterials. While our model is trained to ultimately perform 8 distinct tasks, with available datasets it can be extended to solve additional problems. In a broader sense, this work illustrates a form of multiscale modeling that relates a set of ultimate building blocks (here, byte-level utf8 characters that define the nature of the physical system at hand) to complex output. This materiomic scheme captures complex emergent relationships between universal building block and resulting properties via a synergizing learning capacity to express a set of potentialities embedded in the knowledge used in training, via the interplay of universality and diversity.

### Significance
Predicting the properties of materials based on a flexible description of their structure, environment or process is a long-standing challenge in multiscale modeling. Our MaterioFormer language-model, trained to solve forward and inverse tasks, incorporates a deep learning capacity through attention and graph strategies, to yield a multimodal approach to model and design materials. Since our model is prompt-based and information is encoded consistently via byte-level utf8 tokenization, it can process diverse modalities of information, such as sequence data, description of tasks, and numbers and offers a flexible workflow that integrates human intelligence and AI. Autoregressive training, using pre-training against a large unlabeled dataset, allows for straightforward  adjustment of specific objectives. 

![image](https://github.com/lamm-mit/MateriomicTransformer/assets/101393859/3f40c42f-10e0-496f-b565-773aabc3c4b1)

A deep language model is developed that can solve forward and inverse protein modeling problems. Panel a shows two sample tasks, forward (e.g. calculate secondary structure content of a protein given its sequence) and inverse (design a protein to meet a specified secondary structure content). Overview of the approach implemented, generating molecular structures from amino acid sequences (panel b). The model realizes a variety of calculate and generate tasks to solve multiple protein analysis and design problems. At the heart of the algorithm used here is a text-based transformer architecture that builds interaction graphs using deep multi-headed attention, which serve as the input for a deep graph convolutional neural network to form a nested transformer-graph architecture (panel c). In a broader sense, the modeling conducted here relates an ultimate set of building blocks – here, byte-level utf8 encoded characters – to complex output, which can take many forms. This multiscale scheme captures complex emergent relationships between the basic building block of matter and resulting properties. DSSP is the acronym that refers to the Define Secondary Structure of Proteins (DSSP) algorithm.  

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
![image](https://github.com/lamm-mit/MateriomicTransformer/assets/101393859/c8f3afe6-4c33-47fe-b279-cb80d6dc9cbb)

Overview of the MaterioFormer model, an autoregressive transformer-graph convolutional model built on text-based prompt input for diverse tasks. Panel a depicts details of the implementation of the model, with b showing the causal multi-headed graph self-attention strategy used. The model features a conventional scaled dot-product attention mechanism, using causal self-attention via the triangular mask M, complemented by a graph convolutional neural network. 

### How to set up neural net and sample

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

### Example results

![image](https://github.com/lamm-mit/MateriomicTransformer/assets/101393859/ff41971a-71ab-47cd-8afb-c2cea6fbcb46)

Generative tasks solved after training stage III (see Figure 3 for an overview), showing examples for generating new proteins based on given ratios of secondary structure content. The designed sequences are shown on the left, images of the folded proteins in the center, and a comparison of the design objective (labeled as GT) with the actually obtained secondary structure content (Prediction) shown on the right (for DSSP8 and DSSP3, see Table 1 in paper for definitions). All proteins visualized in this paper are colored (per residue) by the confidence score 50). 

![image](https://github.com/lamm-mit/MateriomicTransformer/assets/101393859/e0d451a3-2400-4432-99d8-56177f7937e6)

Using an amino acid sequence extracted from an existing protein, Sericin 1 (bombyx mori, P07856, SERI1_BOMMO, ser1 gene), and re-engineering the natural protein towards particular design objectives. Herein, panel a shows the original proteins structure and sequence of sericin. Panel b shows a sequence completion task, where the initial sequence is continued in an unconstrained manner. Panel c shows a design task where the design objective is provided alongside the original sequence and then continued to meet the design task. The design task in this case is to generate an alpha-helical protein, which is indeed found towards the end of the protein.  Panels d shows a similar example, however, with the design task to generate a beta-sheet rich protein. This task is more difficult, but after a few trials a solution that meets the design target is obtained. Finally, panel e shows another example where the design task is given is a target with 50% beta-sheet, 20 random coil. This results in a more complex overall protein structure.

```
fname='statedict_V4031.pt'
model.load_state_dict(torch.load(fname))
```
