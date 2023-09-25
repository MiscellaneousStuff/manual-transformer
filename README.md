# manual-transformer

Manual Transformer: Manually programming transformer model weights to determine feasible algorithms

## Algorithms

### Bigram

Manually implement bigram model using `nn.Embedding`.

### Attention Head

Implement same bigram model using attention mechanism.

### Induction Head

Manually implement an induction head. Requires 2 attention
layers stacked on top of each other.

Layer 0:
- Atn
  - Key    (Current token's embed)
  - Query  (Current token's embed)
  - Value  (Token + pos)
  - Output (Token + pos)
- MLP (No activation, identitity function)

Layer 1:
- Atn
  - Key    (Pos + 1, shift using pointer arithmetic on position embed)
  - Query  (Token + pos)
  - Value  (Token + pos)
  - Output (Copy next token along for matched pattern)
- MLP (No activation, identitity function)