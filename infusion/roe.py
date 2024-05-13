import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ROELinear(nn.Linear):
    def __init__(self, length=77, lock=False, is_k = False, *args, **kwargs):
        """
        Args:
            length: length of text encoding.
            lock: apply locking or not.
        """
        super().__init__(*args, **kwargs)
        assert self.bias is None
        self.length = length
        self.lock = lock
        self.is_k = is_k
        self.weight.requires_grad = False
        if not self.lock:
            self.target_output = nn.Parameter(
                torch.empty((self.weight.data.shape[0]), dtype=torch.float32), requires_grad=not lock)
            #self.target_output = nn.Parameter(
            #    torch.empty((2, self.weight.data.shape[0]), dtype=torch.float32), requires_grad=not lock)

    @torch.no_grad()
    def initialize_target_output(self, init_input):
        """
        Args:
            init_input: the encoding of prompt using the super-class word. shape: B x N x D
        """
        if not self.lock:
            self.target_output.data = 0*F.linear(init_input, self.weight).mean(dim=0)[0]
            #self.target_output.data = torch.zeros((self.weight.data.shape[0]), dtype=torch.float32)
        self.init_input = init_input

    def forward(self, input,concept_token_idx, target_input, C_inv, beta=0.75, tau=0.1, input_super=None, **kwargs):
        """
        Args:
            input: the encoding of prompt using the concept word. shape: B x N x D
            target_input: the target input.
            C_inv: the inverse of the uncentered covariance metric.
            beta: bias used in gated rank-1 editing.
            tau: temperature used in gated rank-1 editing.
            input_super: the encoding of prompt using the superclass word.
        """
        # global locking
        if input_super is not None:
            input = input_super

        em_orthogonal_term = input
        batch = input.shape[0]
        W_em_orthogonal_term = F.linear(em_orthogonal_term, self.weight)
        #W_em_orthogonal_term, emb_loss, _ = self.quantize(W_em_orthogonal_term)
        if self.lock:
            h = W_em_orthogonal_term
        #    self.target_output[None, :],emb_loss, _ = self.quantize(self.target_output[None, :])
        else:
            h = W_em_orthogonal_term
            for i in range(concept_token_idx.shape[0]//2):
                if concept_token_idx[i]!=-1:
                    if self.is_k:
                        h[i,int(concept_token_idx[i]+1),:] = h[i,int(concept_token_idx[i]+1),:] + self.target_output
                    else:
                        h[i,int(concept_token_idx[i]+1),:] = h[i,int(concept_token_idx[i]+1),:] + self.target_output
                    #h[i,0,:] = (h[i,0,:] + self.target_output[0])
                    #h[:,2:6,:] = self.target_output[None, :][:,2:6,:] + target_concept[None, :][:,2:6,:]
                    #h[i,:,:] = (target_concept[:,:])
                    #h[i,int(concept_token_idx[i]+1),:] =(self.target_output + target_concept[int(concept_token_idx[i]+1),:])
            #'''
            #W_em_orthogonal_term[:,0,:] = self.target_output[None, :][:,0,:]
            #h =  torch.zeros_like(W_em_orthogonal_term)+self.target_output[None, :]

        return h.to(W_em_orthogonal_term.dtype)


# Inference only
class MultiConceptsROELinear(nn.Linear):
    def __init__(self, length=77, n_concepts=1,lock=False, *args, **kwargs):
        """
        Args:
            length: length of text encoding.
            n_concepts: number of concepts.
        """
        super().__init__(*args, **kwargs)
        assert self.bias is None
        self.length = length
        self.lock = lock
        if not self.lock:
            self.target_outputs = nn.ParameterList([
                nn.Parameter(torch.empty((self.weight.data.shape[0]), dtype=torch.float32))
                for _ in range(n_concepts)
            ])
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, input, concept_token_idx,target_inputs,
                C_inv, beta=0.75, tau=0.1, input_super=None, **kwargs):
        """
        Args:
            input: the encoding of prompt using the concept word. shape: B x N x D
            target_inputs: the target inputs.
            target_inputs_basis: a basis of the spce spanned by target_inputs.
            C_inv: the inverse of the uncentered covariance metric.
            beta: bias used in gated rank-1 editing.
            tau: temperature used in gated rank-1 editing.
            input_super: the encoding of prompt using the superclass word.
        """
        #assert len(target_inputs) == len(self.target_outputs)

        # global locking
        if input_super is not None:
            input = input_super
        em_orthogonal_term = input
        W_em_orthogonal_term = F.linear(em_orthogonal_term, self.weight)
        if self.lock:
            h = W_em_orthogonal_term
        else:
            h = W_em_orthogonal_term
            h[4:8,2,:] = self.target_outputs[0] + h[4:8,2,:]
            #h[:8,16,:] = (self.target_outputs[1] + h[:8,16,:])
            h[4:8,6,:] = self.target_outputs[1] + h[4:8,6,:]
            #h[4:8,24,:] = self.target_outputs[3] + h[4:8,24,:]
        return h.to(W_em_orthogonal_term.dtype)


# Only use when initialization, no weight copy
def roe_to_mc_roe(roe: ROELinear, n_concepts):
    mc_roe = MultiConceptsROELinear(
        n_concepts=n_concepts, bias=False,
        in_features=roe.in_features, out_features=roe.out_features, length=roe.length,lock=roe.lock,
    )
    return mc_roe
