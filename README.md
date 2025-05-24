# Accelerated Hadamard Transform optimized for GPU

AHT: Accelerated Hadamard Transform optimized for GPU. 

## About 

Original implementation for paper: 'BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs'

This implemetation provides batch-processable 'Fast' hadamard transformation using matmul.

## Key features

1. Hadamard transformation implemented as a matrix product, allowing for fast processing on GPUs.

2. If the dimension size of head is a power of 2, any number of potential dimension sizes can be processed.

3. Multiple heads are processed at the same time.

***

AHT Matrix is below.
AHT Matrix generation by Sylvester construction

```math
\displaylines{
H_1 = \begin{bmatrix} 1 \end{bmatrix} \\
H_{2n} = \frac{1}{\sqrt{2}} \begin{bmatrix}
H_n & H_n \\
H_n & -H_n
\end{bmatrix}
}
```

## Implementation and License

This repository is official pure pytorch implementation.

Licensed under ["MIT License"](https://mit-license.org/).

Commercial use permitted

## How to use

- Clone the repository

```bash
git clone https://github.com/Rikka-Botan/Accelerated_Hadamard.git
```

- Model create

```python
"""
Args:
hadamard_size: int - model head dim
"""

from model.AHT_modeling import BotanAHT

model = BotanAHT(
  hadamard_size
)
output = model(hidden_states)
```

## Acknowledgements

I thank the developers of python and pytorch.

I thank all the researchers for their efforts to date.

I thank Japan's high standard of education.

And most of all, thank you for your interest in this repository.

## Citations

I would be happy to include a citation at the end, but it is not required.

Feel free to use this model.


## Contact Us

[My X account](https://x.com/peony__snow)


## About Author

### Rikka Botan

Japanese independent researcher having shy and pampered personality >_<

Twin-tail hair is a charm point :)

Interested in natural language processings. 

Usually using python and C.

![RikkaBotan_Logo](https://github.com/user-attachments/assets/92913f91-9136-4d44-8b4d-8a2120118a05)
