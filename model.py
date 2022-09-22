# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 17:28:15 2022

@author: abood
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

class Projection(keras.layers.Layer):
    '''
    Custom layer for dividing image into patches and implementing linear projection step.
    
    * Input
    * Conv2D
    * Reshape
    
    Parameters
    =========
    
    img_size: int
        Hight (or width) of the image [assumed to be squared but doesn't have to be]
                                       
    embed_size: int
        Embedding size.
        
    patch_size: int
        Size of patch the images with be split into.
    
    Inputs
    =======
    
    input shape: shape
        [batch_size, Hight, Width, Channels]
        
    Outputs
    =======
    
    Output shape: shape
        [batch_size, n_patches, embed_size]
    '''
    def __init__(self, embed_size, patch_size, img_size):
        
        super(Projection, self).__init__()
        self.embed_size = embed_size
        self.patch_size = patch_size
        self.n_patches = (img_size // self.patch_size) ** 2 # N = HW/P^2
        
    def build(self, input_shape):
        
        
        self.conv = keras.layers.Conv2D(filters = self.embed_size, 
                                        kernel_size = self.patch_size, 
                                        strides = self.patch_size,
                           input_shape = input_shape[1:])
        
    def call(self,  inputs):
        x = inputs
        x = self.conv(x) # shape --> [batch_size, n_patches ** 0.5, n_patches ** 0.5, embed_size]
        
        shape = [keras.backend.shape(x)[k] for k in range(4)]
        x = tf.reshape(x, [shape[0], shape[1]*shape[2], shape[3]]) # reshape to [batch_size, n_patches, embed_size]
        
        return x

class Attention(keras.layers.Layer):
    '''
    Class Implementing multi-head self attention 
    
    * head = softmax(q @ k_t // scale) @ v
    * concat(heads)
    * Linear Projection
    
    parameters
    ==========
    
    embed_dim: int
        Embedding size.
        
    n_head: int
        Number of heads of the multi-head self attention.
        
    Input
    =====
    
    input shape: shape
        [batch_size, n_patches + 1, embed_size]
       
    Output
    =====
    
    output shape: shape
        [batch_size, n_patches + 1, embed_size]
    '''
    def __init__(self, embed_size, n_heads):
        super(Attention, self).__init__()
        
        self.n_heads = n_heads
        self.head_dim = embed_size // n_heads # when concatenated will result in embed_size
        self.scale = self.head_dim ** (-0.5)
        
        self.query = keras.layers.Dense(
            self.head_dim
        )
        self.key = keras.layers.Dense(
            self.head_dim
        )
        self.value = keras.layers.Dense(
            self.head_dim
        )
        self.softmax = keras.layers.Softmax()
        
        self.proj = keras.layers.Dense(embed_size)
        
    def call(self, inputs):
        
        q, k, v = self.query(inputs), self.key(inputs), self.value(inputs)
        k_t = tf.transpose(k, perm=[0, 2, 1]) # Transpose --> [batch_size, head_dim, n_patches + 1]
        
        attn_filter = (q @ k_t) * self.scale
        attn_filter = self.softmax(attn_filter)
        
        attn_head = attn_filter @ v
        attn_head = tf.expand_dims(attn_head, axis = 0) # [1, batch_size, n_patches + 1, head_dim]

        heads = tf.concat([attn_head for _ in range(self.n_heads)], axis = 0)  # [n_heads, batch_size, n_patches + 1, head_dim]
        heads = tf.transpose(heads, perm=[1, 2, 3, 0]) # [batch_size, n_patches + 1, head_dim, n_heads]

        bs, n_p, hd, nh= [keras.backend.shape(heads)[k] for k in range(4)]
        heads = tf.reshape(heads, [bs, n_p, hd * nh]) # [batch_size, n_patches + 1, embed_dim]
        
        return self.proj(heads)

class MLP(keras.layers.Layer):
    '''
    Class Implementing FeedForward Layer.
    
    * Linear
    * Activation (GELU)
    * Linear
    
    parameters
    ==========
    
    embed_size: int
        Embedding size.
    
    hidden_size: int
        output dim of first hidden layer
    
    activation_fn: str
        activation function applied after the first hidden layer
        
    Input
    =====
    
    input shape: shape
        [batch_size, n_patches + 1, embed_size]
       
    Output
    =====
    
    output shape: shape
        [batch_size, n_patches + 1, embed_size]
    '''
    def __init__(self, embed_size, hidden_size, activation_fn = 'gelu'):
        super(MLP, self).__init__()
        
        self.Hidden = keras.layers.Dense(
            hidden_size
        )
        self.activation = keras.activations.get(activation_fn)
        self.Linear = keras.layers.Dense(
            embed_size
        )
    def call(self, inputs):
        
        x = inputs
        x = self.Hidden(x)
        x = self.activation(x)
        
        return self.Linear(x)

class TransformerEncoder(keras.layers.Layer):
    '''
    Class for implementing Transformer Encoder Block.
    
    * Input
    * LayerNorm
    * Multi-head self attention
    * residual connection
    * LayerNorm
    * Multi-layer perceptron
    * residual connection
    
    parameters
    ==========
    
    embed_size: int
        Embedding size.
    
    n_heads: int
        
    mlpHidden_size: int
        output dim of first hidden layer of the MLP
    
    mlp_activation: str
        activation function applied after the first hidden layer of the MLP
        
    Input
    =====
    
    input shape: shape
        [batch_size, n_patches + 1, embed_size]
       
    Output
    =====
    
    output shape: shape
        [batch_size, n_patches + 1, embed_size]
    '''
    def __init__(self, embed_size, n_heads, mlpHidden_size, mlp_activation):
        super(TransformerEncoder, self).__init__()
        
        self.norm1 = keras.layers.LayerNormalization()
        self.MSA = Attention(embed_size, n_heads)
        self.MLP = MLP(embed_size, mlpHidden_size, mlp_activation)
        self.norm2 = keras.layers.LayerNormalization()
        
    def call(self, inputs):
        
        x = inputs
        
        x = self.norm1(x)
        x = self.MSA(x)
        x = x + inputs
        
        y = x
        
        y = self.norm2(y)
        y = self.MLP(y)
        y = y + x
        
        return y
    
class VisionTransformer(keras.Model):
    '''
    Class for implementing Vision Transformer Architecture.
    
    * Input
    * Linear Projection
    * prepend cls token then add positional embedding
    * transformer encoder
    * LayerNorm
    * Linear
    
    parameters
    ==========
    
    embed_size: int
        Embedding size.
    
    patch_size: int
        Size of patch the images with be split into.
    
    n_head: int
        Number of heads of the multi-head self attention.
    
    mlpHidden_size: int
        output dim of first hidden layer of the MLP
    
    mlp_activation: str
        activation function applied after the first hidden layer of the MLP
    
    n_blocks: int
        Number of transformer encoder block
        
    n_classes: int
        Number of class for our image classification problem
        
    Input
    =====
    
    input shape: shape
        [batch_size, Hight, Width, Channels]
       
    Output
    =====
    
    output shape: shape
        [batch_size, n_classes]
    '''
    def __init__(self, embed_size,
                 patch_size,
                 n_heads, 
                 mlpHidden_size, 
                 mlp_activation,
                n_blocks,
                n_classes,
                img_size = 32,
                batch_size = 32):
        super(VisionTransformer, self).__init__()
        
        self.embed_size = embed_size
        self.proj = Projection(embed_size, patch_size, img_size)
        self.cls_token = tf.Variable(tf.zeros(shape = [1, 1, embed_size])) # Will be broadcasted to batch size
        self.pos_embed = tf.Variable(tf.zeros(shape = [1, self.proj.n_patches + 1, embed_size]))
        
        self.Encoder_blocks = keras.Sequential([
            TransformerEncoder(embed_size, 
                               n_heads, 
                               mlpHidden_size, 
                               mlp_activation
        ) for _ in range(n_blocks)
        ])
        
        self.norm = keras.layers.LayerNormalization()
        self.Linear = keras.layers.Dense(n_classes, activation  = 'softmax')
        
   
    
    def call(self, inputs):
        batch_size, hight, width, channels = inputs.shape
        
        linear_embed = self.proj(inputs) # shape --> [batch_size, n_patches, embed_size]
        
        broadcast_shape = tf.where([True, False, False],
                           keras.backend.shape(tf.expand_dims(linear_embed[:, 0], axis = 1)), [0, 1, self.embed_size]) # for broadcasting to a dynamic shape [None,  1, embed_size]
        cls_token = tf.broadcast_to(self.cls_token, shape = broadcast_shape) #Found solution here --> (https://stackoverflow.com/questions/63211206/how-to-broadcast-along-batch-dimension-with-tensorflow-functional-api)
        
        assert cls_token.shape[0] == linear_embed.shape[0]
        linear_proj = tf.concat([cls_token, linear_embed], axis = 1) # shape --> [batch_size, n_patches + 1, embed_size]
        
        x = self.Encoder_blocks(linear_proj + self.pos_embed)
        x = self.norm(x)
        
        cls_token_final = x[:, 0] # Only the output of the cls token should be considered
        return self.Linear(cls_token_final)


##################################################################################
############################## TEST ##############################################
##################################################################################


if __name__ == "__main__":
    rnd_img = tf.random.uniform(shape = [16, 32, 32, 3], dtype = tf.float32) # shape --> [batch_size, Hight, Width, Channels]
    
    model = VisionTransformer(embed_size = 512,
                     patch_size = 16,
                     n_heads = 6, 
                     mlpHidden_size = 1024, 
                     mlp_activation = 'gelu',
                    n_blocks = 3,
                    n_classes = 10)
    
    output = model(rnd_img) # # shape --> [batch_size, n_classes]
