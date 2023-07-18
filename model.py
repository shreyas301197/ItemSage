import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ItemSageModel(nn.Module):
    
    def __init__(self, nhead = 8,nlayers = 1,**kwargs):
        super(ItemSageModel,self).__init__()
        
        home_feat_dim = kwargs['home_features_dim']
        image_embedding_dim = kwargs['image_embed_dim']
        text_embedding_dim = kwargs['text_embed_dim']
        skg_embedding_dim = kwargs['skg_embed_dim']
        ps_embedding_dim = kwargs['ps_embed_dim']
        transformer_input_dim = kwargs['transformer_input_dim']
        output_dim = kwargs['final_embed_dim']
        n_zips = kwargs['n_zips']
        zip_embed_dim = kwargs['zip_embed_dim']
        
        if kwargs['loss_margin'] :
            self.loss = kwargs['loss'](margin=kwargs['loss_margin']) # criterion created
        else:
            self.loss = kwargs['loss']( )
            
        self.device = kwargs['device']
        
        # converting all input to the same dimensionality for forming transformer input seq
        self.zip_embed = nn.Embedding(n_zips,zip_embed_dim)
        self.home_linear = nn.Linear(home_feat_dim+zip_embed_dim, transformer_input_dim)
        self.image_linear = nn.Linear(image_embedding_dim, transformer_input_dim)
        self.text_linear = nn.Linear(text_embedding_dim, transformer_input_dim)
        self.skg_linear = nn.Linear(skg_embedding_dim, transformer_input_dim)
        self.ps_linear = nn.Linear(ps_embedding_dim, transformer_input_dim)
        
        # transforming the CLS token
        self.global_CLS_linear = nn.Linear(1, transformer_input_dim)
        
        # Transformer Encoder Layer
        self.encoder_layer = TransformerEncoderLayer(d_model=transformer_input_dim, nhead=nhead,\
                                                     batch_first=True,norm_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        
        # MLP head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(transformer_input_dim),
            nn.Linear(transformer_input_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, output_dim)
        )
    
    # option to use one of the two loss types
    def _contrastive_loss(self, x1,x2,label):
        return self.loss(x1,x2,label)
    
    def _classification_loss(self,score,label):
        return self.loss(score,label)

    # itemsage tower
    def _encoder(self, text_embeddings, image_embeddings, skg_embeddings, ps_embeddings, home_features, zip_idx):
        """ 
        :param image_embeddings: (bsz, seq_length, D_img) - 
        :param text_embeddings: (bsz,  D_text) - make seq_length = 1 for openAI embeddings - using unsqueeze()
        :param skg_embeddings: (bsz, D_skg) - make seq_length = 1 because 1 embedding/listing - using unsqueeze()
        :param ps_embeddings: (bsz, D_ps) - make seq_length = 1 because 1 embedding/listing - using unsqueeze()
        :param home_features: (bsz, D_home_feat) - make seq_length = 1 because 1 embedding/listing - using unsqueeze()
        """
        transformed_image_embeddings = self.image_linear(image_embeddings)
        transformed_text_embeddings = self.text_linear(torch.unsqueeze(text_embeddings,1))
        transformed_skg_embeddings = self.skg_linear(torch.unsqueeze(skg_embeddings,1))
        transformed_ps_embeddings = self.ps_linear(torch.unsqueeze(ps_embeddings,1))
        zip_embedding = self.zip_embed(zip_idx)
        home_features = torch.cat((home_features,zip_embedding),dim=1)
        transformed_home_feat = self.home_linear(torch.unsqueeze(home_features,1))
        
        # Apply linear transformation to the global token
        batch_size = transformed_image_embeddings.size(0)
        cls_token = torch.ones(batch_size, 1,1).to(self.device)

        
        # Apply linear transformation to make [CLS] token 512-dimensional
        cls_token = self.global_CLS_linear(cls_token)
        
        # print(cls_token.shape, transformed_image_embeddings.shape, transformed_text_embeddings.shape)
        # embedding sequence
        embedding_seq = torch.cat((cls_token, transformed_image_embeddings, transformed_text_embeddings,\
                                   transformed_skg_embeddings,transformed_ps_embeddings,transformed_home_feat), dim=1)
        # print('embedding_seq shape : ',embedding_seq.shape)
        transformer_output = self.transformer_encoder(embedding_seq)
        # print('transformer_output shape : ',transformer_output.shape)
        cls_output = transformer_output[:, 0, :]
        product_embedding = self.mlp_head(cls_output)
        return product_embedding
    
    
    def forward(self,**kwargs1):
        anchor_encoder = self._encoder(kwargs1['anchor'][0],kwargs1['anchor'][1],kwargs1['anchor'][2],\
                                      kwargs1['anchor'][3],kwargs1['anchor'][4],kwargs1['anchor'][5]) # query tower
        
        engaged_encoder = self._encoder(kwargs1['engaged'][0],kwargs1['engaged'][1],kwargs1['engaged'][2],\
                                       kwargs1['engaged'][3],kwargs1['engaged'][4],kwargs1['engaged'][5]) # example tower
        
        # normalize both embeddings so that cosine_similarity == dot_product
        return F.normalize(anchor_encoder,dim=1),F.normalize(engaged_encoder,dim=1)
    